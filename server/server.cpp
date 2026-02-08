// server.cpp - Minimal Distributed LLM Param/Grad Server (cross-platform)
// - HTTP/1.1 (Content-Length only; no chunked encoding)
// - Model params served per-tensor with optional range slicing
// - Sparse gradient submission via binary packet (indices + values)
// - Lazy sparse AdamW (optimizer state stored only for touched indices)
//
// Build (Linux/macOS):
//   g++ -O2 -std=c++17 -pthread server.cpp -o server
//
// Build (Windows MSVC Developer Prompt):
//   cl /O2 /std:c++17 server.cpp Ws2_32.lib
//
// Run:
//   ./server 8080
//   ./server 8080 --tiny

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <queue>
#include <random>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")
typedef int socklen_t;
#define close_socket closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int SOCKET;
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define close_socket close
#endif

namespace fs = std::filesystem;

// ============================================================================
// Small utilities
// ============================================================================

static inline std::string to_lower(std::string s) {
  for (char &c : s) {
    c = (char)std::tolower((unsigned char)c);
  }
  return s;
}

static std::string mime_type_from_path(const std::string &path) {
  auto dot = path.find_last_of('.');
  std::string ext = (dot == std::string::npos) ? "" : path.substr(dot + 1);
  ext = to_lower(ext);

  if (ext == "html" || ext == "htm") {
    return "text/html; charset=utf-8";
  }
  if (ext == "css") {
    return "text/css; charset=utf-8";
  }
  if (ext == "js") {
    return "application/javascript; charset=utf-8";
  }
  if (ext == "json") {
    return "application/json; charset=utf-8";
  }
  if (ext == "txt") {
    return "text/plain; charset=utf-8";
  }
  if (ext == "png") {
    return "image/png";
  }
  if (ext == "jpg" || ext == "jpeg") {
    return "image/jpeg";
  }
  if (ext == "svg") {
    return "image/svg+xml";
  }
  if (ext == "ico") {
    return "image/x-icon";
  }
  return "application/octet-stream";
}

// Very small "good enough" path hardening:
// - no ".."
// - no backslashes
// - no ':' (Windows drive letter / scheme-like)
static bool is_safe_rel_path(const std::string &rel) {
  if (rel.find("..") != std::string::npos) {
    return false;
  }
  if (rel.find('\\') != std::string::npos) {
    return false;
  }
  if (rel.find(':') != std::string::npos) {
    return false;
  }
  if (rel.find('\0') != std::string::npos) {
    return false;
  }
  return true;
}

static std::string read_file_to_string(const std::string &path) {
  std::ifstream f(path, std::ios::in | std::ios::binary);
  if (!f.is_open())
    return "";
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

static std::string json_get_val(const std::string &json,
                                const std::string &key) {
  std::string search = "\"" + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos)
    return "";
  pos += search.size();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                               json[pos] == '\n' || json[pos] == '\r'))
    pos++;
  size_t end = pos;
  if (pos < json.size() && json[pos] == '"') {
    pos++;
    end = json.find('"', pos);
    if (end == std::string::npos)
      return "";
    return json.substr(pos, end - pos);
  } else {
    while (end < json.size() && json[end] != ',' && json[end] != '}' &&
           json[end] != '\n' && json[end] != '\r' && json[end] != ' ' &&
           json[end] != '\t')
      end++;
    return json.substr(pos, end - pos);
  }
}

// ============================================================================
// Logging
// ============================================================================

enum class LogLevel { INFO, WARN, ERR, DEBUG };

static const char *log_level_str(LogLevel l) {
  switch (l) {
  case LogLevel::INFO:
    return "INFO";
  case LogLevel::WARN:
    return "WARN";
  case LogLevel::ERR:
    return "ERROR";
  case LogLevel::DEBUG:
    return "DEBUG";
  default:
    return "???";
  }
}

static std::string current_time_str() {
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %H:%M:%S");
  return ss.str();
}

static std::mutex g_log_mtx;

class Log {
public:
  explicit Log(LogLevel l, const std::string &comp) : level_(l), comp_(comp) {
    os_ << "[" << current_time_str() << "] [" << log_level_str(level_) << "] ["
        << comp_ << "] ";
  }
  ~Log() {
    std::lock_guard<std::mutex> lock(g_log_mtx);
    std::cerr << os_.str() << std::endl;
  }
  template <typename T> Log &operator<<(const T &val) {
    os_ << val;
    return *this;
  }

private:
  LogLevel level_;
  std::string comp_;
  std::ostringstream os_;
};

#define LOG_INFO(c) Log(LogLevel::INFO, c)
#define LOG_WARN(c) Log(LogLevel::WARN, c)
#define LOG_ERROR(c) Log(LogLevel::ERR, c)
#define LOG_DEBUG(c) Log(LogLevel::DEBUG, c)

// ============================================================================
// Network utilities
// ============================================================================

static bool send_all(SOCKET s, const uint8_t *data, size_t len) {
  size_t sent = 0;
  while (sent < len) {
#ifdef _WIN32
    int n = ::send(s, (const char *)(data + sent), (int)(len - sent), 0);
#else
    int n = (int)::send(s, (const char *)(data + sent), len - sent, 0);
#endif
    if (n <= 0) {
      return false;
    }
    sent += (size_t)n;
  }
  return true;
}

static bool send_all(SOCKET s, const std::string &str) {
  return send_all(s, (const uint8_t *)str.data(), str.size());
}

// ============================================================================
// HTTP (minimal)
// ============================================================================

struct HttpRequest {
  std::string method;
  std::string target;                                   // includes query
  std::string path;                                     // without query
  std::string query;                                    // without '?'
  std::unordered_map<std::string, std::string> headers; // lower-case keys
  std::vector<uint8_t> body;
};

struct HttpResponse {
  int status = 200;
  std::string status_text = "OK";
  std::string content_type = "text/plain; charset=utf-8";
  std::vector<std::pair<std::string, std::string>> headers;
  std::vector<uint8_t> body;

  static HttpResponse json(const std::string &s, int code = 200) {
    HttpResponse r;
    r.status = code;
    r.status_text = (code == 200) ? "OK" : "Error";
    r.content_type = "application/json; charset=utf-8";
    r.body.assign(s.begin(), s.end());
    return r;
  }

  static HttpResponse text(const std::string &s, int code = 200) {
    HttpResponse r;
    r.status = code;
    r.status_text = (code == 200) ? "OK" : "Error";
    r.content_type = "text/plain; charset=utf-8";
    r.body.assign(s.begin(), s.end());
    return r;
  }

  static HttpResponse bin(std::vector<uint8_t> b,
                          const std::string &ctype = "application/octet-stream",
                          int code = 200) {
    HttpResponse r;
    r.status = code;
    r.status_text = (code == 200) ? "OK" : "Error";
    r.content_type = ctype;
    r.body = std::move(b);
    return r;
  }
};

static std::string http_serialize_headers(const HttpResponse &res) {
  std::ostringstream ss;
  ss << "HTTP/1.1 " << res.status << " " << res.status_text << "\r\n";
  ss << "Content-Type: " << res.content_type << "\r\n";
  ss << "Content-Length: " << res.body.size() << "\r\n";
  ss << "Access-Control-Allow-Origin: *\r\n";
  ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
  ss << "Access-Control-Allow-Headers: Content-Type\r\n";
  ss << "Connection: close\r\n";
  for (const auto &kv : res.headers) {
    ss << kv.first << ": " << kv.second << "\r\n";
  }
  ss << "\r\n";
  return ss.str();
}

static bool read_http_request(SOCKET s, HttpRequest &out, std::string &err,
                              size_t max_body_bytes = (256ull << 20)) {
  std::string buf;
  buf.reserve(8192);

  auto recv_some = [&](char *tmp, size_t cap) -> int {
#ifdef _WIN32
    return ::recv(s, tmp, (int)cap, 0);
#else
    return (int)::recv(s, tmp, cap, 0);
#endif
  };

  // Read until headers end
  char tmp[16384];
  size_t header_end = std::string::npos;
  while (true) {
    int n = recv_some(tmp, sizeof(tmp));
    if (n <= 0) {
      err = "recv failed";
      return false;
    }
    buf.append(tmp, tmp + n);
    header_end = buf.find("\r\n\r\n");
    if (header_end != std::string::npos) {
      break;
    }
    if (buf.size() > (1ull << 20)) {
      err = "headers too large";
      return false;
    }
  }

  std::string header_block = buf.substr(0, header_end);
  std::string remaining = buf.substr(header_end + 4);

  std::istringstream hs(header_block);
  std::string req_line;
  if (!std::getline(hs, req_line)) {
    err = "bad request line";
    return false;
  }
  if (!req_line.empty() && req_line.back() == '\r') {
    req_line.pop_back();
  }

  {
    std::istringstream rl(req_line);
    rl >> out.method >> out.target;
  }
  if (out.method.empty() || out.target.empty()) {
    err = "bad request line tokens";
    return false;
  }

  // Split path/query
  {
    auto qpos = out.target.find('?');
    if (qpos == std::string::npos) {
      out.path = out.target;
      out.query = "";
    } else {
      out.path = out.target.substr(0, qpos);
      out.query = out.target.substr(qpos + 1);
    }
  }

  // Headers
  out.headers.clear();
  std::string line;
  size_t content_length = 0;
  while (std::getline(hs, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    std::string key = to_lower(line.substr(0, pos));
    std::string val = line.substr(pos + 1);
    while (!val.empty() && (val[0] == ' ' || val[0] == '\t')) {
      val.erase(val.begin());
    }
    out.headers[key] = val;
    if (key == "content-length") {
      try {
        content_length = (size_t)std::stoull(val);
      } catch (...) {
        err = "invalid content-length";
        return false;
      }
    }
  }

  if (content_length > max_body_bytes) {
    err = "body too large";
    return false;
  }

  out.body.clear();
  out.body.reserve(content_length);

  // If body already partially read
  if (!remaining.empty()) {
    size_t take = std::min(remaining.size(), content_length);
    out.body.insert(out.body.end(), remaining.begin(),
                    remaining.begin() + take);
  }

  while (out.body.size() < content_length) {
    int n = recv_some(tmp, sizeof(tmp));
    if (n <= 0) {
      err = "recv body failed";
      return false;
    }
    size_t need = content_length - out.body.size();
    size_t take = std::min((size_t)n, need);
    out.body.insert(out.body.end(), (uint8_t *)tmp, (uint8_t *)tmp + take);
  }

  return true;
}

static std::unordered_map<std::string, std::string>
parse_query_kv(const std::string &q) {
  std::unordered_map<std::string, std::string> m;
  size_t i = 0;
  while (i < q.size()) {
    size_t amp = q.find('&', i);
    if (amp == std::string::npos) {
      amp = q.size();
    }
    std::string part = q.substr(i, amp - i);
    size_t eq = part.find('=');
    if (eq == std::string::npos) {
      m[part] = "";
    } else {
      m[part.substr(0, eq)] = part.substr(eq + 1);
    }
    i = amp + 1;
  }
  return m;
}

// ============================================================================
// Model / Tensors
// ============================================================================

struct Tensor {
  std::vector<float> data;
  std::vector<int> shape;
  int64_t size = 0;

  void save(std::ostream &os) const {
    os.write((const char *)data.data(), data.size() * sizeof(float));
  }
  void load(std::istream &is) {
    is.read((char *)data.data(), data.size() * sizeof(float));
  }

  Tensor() = default;
  explicit Tensor(const std::vector<int> &s) : shape(s) {
    size = 1;
    for (int d : shape) {
      size *= d;
    }
    data.assign((size_t)size, 0.0f);
  }
};

struct ModelConfig {
  int vocab_size = 50257;
  int d_model = 768;
  int n_heads = 12;
  int n_layers = 12;
  int d_ff = 3072;
  int max_seq_len = 512;

  void load_json(const std::string &s) {
    auto v = [&](const char *k, int &out) {
      std::string val = json_get_val(s, k);
      if (!val.empty())
        out = std::stoi(val);
    };
    v("vocab_size", vocab_size);
    v("d_model", d_model);
    v("n_heads", n_heads);
    v("n_layers", n_layers);
    v("d_ff", d_ff);
    v("max_seq_len", max_seq_len);
  }
};

struct TrainConfig {
  float learning_rate = 3e-4f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  float weight_decay = 0.01f;
  int min_nodes_for_update = 2;

  void load_json(const std::string &s) {
    auto v = [&](const char *k, float &out) {
      std::string val = json_get_val(s, k);
      if (!val.empty())
        out = std::stof(val);
    };
    v("learning_rate", learning_rate);
    v("beta1", beta1);
    v("beta2", beta2);
    v("eps", eps);
    v("weight_decay", weight_decay);
    std::string mn = json_get_val(s, "min_nodes_for_update");
    if (!mn.empty())
      min_nodes_for_update = std::stoi(mn);
  }
};

struct TensorMeta {
  uint32_t id = 0;
  std::string name;
  std::vector<int> shape;
  int64_t elements = 0;
};

struct ModelParams {
  ModelConfig cfg;

  Tensor token_embed;
  Tensor pos_embed;

  std::vector<Tensor> ln1_gamma, ln1_beta;
  std::vector<Tensor> qkv_weight, qkv_bias;
  std::vector<Tensor> proj_weight, proj_bias;

  std::vector<Tensor> ln2_gamma, ln2_beta;
  std::vector<Tensor> ffn_up, ffn_up_bias;
  std::vector<Tensor> ffn_down, ffn_down_bias;

  Tensor ln_f_gamma, ln_f_beta;
  Tensor lm_head;

  explicit ModelParams(ModelConfig c) : cfg(std::move(c)) { init(); }

  void init() {
    token_embed = Tensor({cfg.vocab_size, cfg.d_model});
    pos_embed = Tensor({cfg.max_seq_len, cfg.d_model});

    ln1_gamma.assign(cfg.n_layers, Tensor({cfg.d_model}));
    ln1_beta.assign(cfg.n_layers, Tensor({cfg.d_model}));
    qkv_weight.assign(cfg.n_layers, Tensor({3 * cfg.d_model, cfg.d_model}));
    qkv_bias.assign(cfg.n_layers, Tensor({3 * cfg.d_model}));
    proj_weight.assign(cfg.n_layers, Tensor({cfg.d_model, cfg.d_model}));
    proj_bias.assign(cfg.n_layers, Tensor({cfg.d_model}));

    ln2_gamma.assign(cfg.n_layers, Tensor({cfg.d_model}));
    ln2_beta.assign(cfg.n_layers, Tensor({cfg.d_model}));
    ffn_up.assign(cfg.n_layers, Tensor({cfg.d_ff, cfg.d_model}));
    ffn_up_bias.assign(cfg.n_layers, Tensor({cfg.d_ff}));
    ffn_down.assign(cfg.n_layers, Tensor({cfg.d_model, cfg.d_ff}));
    ffn_down_bias.assign(cfg.n_layers, Tensor({cfg.d_model}));

    ln_f_gamma = Tensor({cfg.d_model});
    ln_f_beta = Tensor({cfg.d_model});
    lm_head = Tensor({cfg.vocab_size, cfg.d_model});

    std::mt19937 gen(42);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    auto init_n = [&](Tensor &t) {
      for (auto &v : t.data) {
        v = nd(gen);
      }
    };

    init_n(token_embed);
    init_n(pos_embed);

    for (int l = 0; l < cfg.n_layers; l++) {
      std::fill(ln1_gamma[l].data.begin(), ln1_gamma[l].data.end(), 1.0f);
      std::fill(ln2_gamma[l].data.begin(), ln2_gamma[l].data.end(), 1.0f);
      init_n(qkv_weight[l]);
      init_n(proj_weight[l]);
      init_n(ffn_up[l]);
      init_n(ffn_down[l]);
    }

    std::fill(ln_f_gamma.data.begin(), ln_f_gamma.data.end(), 1.0f);
    init_n(lm_head);
  }

  int64_t total_params() const {
    int64_t t = 0;
    auto add = [&](const Tensor &x) { t += x.size; };
    add(token_embed);
    add(pos_embed);
    for (int l = 0; l < cfg.n_layers; l++) {
      add(ln1_gamma[l]);
      add(ln1_beta[l]);
      add(qkv_weight[l]);
      add(qkv_bias[l]);
      add(proj_weight[l]);
      add(proj_bias[l]);
      add(ln2_gamma[l]);
      add(ln2_beta[l]);
      add(ffn_up[l]);
      add(ffn_up_bias[l]);
      add(ffn_down[l]);
      add(ffn_down_bias[l]);
    }
    add(ln_f_gamma);
    add(ln_f_beta);
    add(lm_head);
    return t;
  }

  void save(std::ostream &os) const {
    token_embed.save(os);
    pos_embed.save(os);
    for (int l = 0; l < cfg.n_layers; l++) {
      ln1_gamma[l].save(os);
      ln1_beta[l].save(os);
      qkv_weight[l].save(os);
      qkv_bias[l].save(os);
      proj_weight[l].save(os);
      proj_bias[l].save(os);
      ln2_gamma[l].save(os);
      ln2_beta[l].save(os);
      ffn_up[l].save(os);
      ffn_up_bias[l].save(os);
      ffn_down[l].save(os);
      ffn_down_bias[l].save(os);
    }
    ln_f_gamma.save(os);
    ln_f_beta.save(os);
    lm_head.save(os);
  }

  void load(std::istream &is) {
    token_embed.load(is);
    pos_embed.load(is);
    for (int l = 0; l < cfg.n_layers; l++) {
      ln1_gamma[l].load(is);
      ln1_beta[l].load(is);
      qkv_weight[l].load(is);
      qkv_bias[l].load(is);
      proj_weight[l].load(is);
      proj_bias[l].load(is);
      ln2_gamma[l].load(is);
      ln2_beta[l].load(is);
      ffn_up[l].load(is);
      ffn_up_bias[l].load(is);
      ffn_down[l].load(is);
      ffn_down_bias[l].load(is);
    }
    ln_f_gamma.load(is);
    ln_f_beta.load(is);
    lm_head.load(is);
  }

  // Stable tensor ID ordering.
  std::vector<TensorMeta> manifest() const {
    std::vector<TensorMeta> out;
    uint32_t id = 0;

    auto push = [&](const std::string &name, const Tensor &t) {
      TensorMeta m;
      m.id = id++;
      m.name = name;
      m.shape = t.shape;
      m.elements = t.size;
      out.push_back(std::move(m));
    };

    push("token_embed", token_embed);
    push("pos_embed", pos_embed);

    for (int l = 0; l < cfg.n_layers; l++) {
      push("layers." + std::to_string(l) + ".ln1_gamma", ln1_gamma[l]);
      push("layers." + std::to_string(l) + ".ln1_beta", ln1_beta[l]);
      push("layers." + std::to_string(l) + ".qkv_weight", qkv_weight[l]);
      push("layers." + std::to_string(l) + ".qkv_bias", qkv_bias[l]);
      push("layers." + std::to_string(l) + ".proj_weight", proj_weight[l]);
      push("layers." + std::to_string(l) + ".proj_bias", proj_bias[l]);

      push("layers." + std::to_string(l) + ".ln2_gamma", ln2_gamma[l]);
      push("layers." + std::to_string(l) + ".ln2_beta", ln2_beta[l]);
      push("layers." + std::to_string(l) + ".ffn_up", ffn_up[l]);
      push("layers." + std::to_string(l) + ".ffn_up_bias", ffn_up_bias[l]);
      push("layers." + std::to_string(l) + ".ffn_down", ffn_down[l]);
      push("layers." + std::to_string(l) + ".ffn_down_bias", ffn_down_bias[l]);
    }

    push("ln_f_gamma", ln_f_gamma);
    push("ln_f_beta", ln_f_beta);
    push("lm_head", lm_head);

    return out;
  }

  // Returns pointer to tensor by ID (same order as manifest()).
  Tensor *tensor_by_id(uint32_t id) {
    uint32_t cur = 0;

    auto take = [&](Tensor &t) -> Tensor * {
      if (cur == id) {
        return &t;
      }
      cur++;
      return nullptr;
    };

    if (auto p = take(token_embed)) {
      return p;
    }
    if (auto p = take(pos_embed)) {
      return p;
    }

    for (int l = 0; l < cfg.n_layers; l++) {
      if (auto p = take(ln1_gamma[l])) {
        return p;
      }
      if (auto p = take(ln1_beta[l])) {
        return p;
      }
      if (auto p = take(qkv_weight[l])) {
        return p;
      }
      if (auto p = take(qkv_bias[l])) {
        return p;
      }
      if (auto p = take(proj_weight[l])) {
        return p;
      }
      if (auto p = take(proj_bias[l])) {
        return p;
      }

      if (auto p = take(ln2_gamma[l])) {
        return p;
      }
      if (auto p = take(ln2_beta[l])) {
        return p;
      }
      if (auto p = take(ffn_up[l])) {
        return p;
      }
      if (auto p = take(ffn_up_bias[l])) {
        return p;
      }
      if (auto p = take(ffn_down[l])) {
        return p;
      }
      if (auto p = take(ffn_down_bias[l])) {
        return p;
      }
    }

    if (auto p = take(ln_f_gamma)) {
      return p;
    }
    if (auto p = take(ln_f_beta)) {
      return p;
    }
    if (auto p = take(lm_head)) {
      return p;
    }

    return nullptr;
  }
};

// ============================================================================
// FP16 conversion (IEEE 754 half) for transport
// ============================================================================

static uint16_t fp16_from_fp32(float f) {
  uint32_t x;
  std::memcpy(&x, &f, sizeof(x));

  uint32_t sign = (x >> 31) & 0x1;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127;
  uint32_t mant = x & 0x7FFFFF;

  uint16_t hsign = (uint16_t)(sign << 15);

  if (((x >> 23) & 0xFF) == 0xFF) { // Inf/NaN
    uint16_t hexp = 0x1F << 10;
    uint16_t hmant = (mant ? 0x200 : 0);
    return (uint16_t)(hsign | hexp | hmant);
  }

  if (exp > 15) { // overflow -> inf
    return (uint16_t)(hsign | (0x1F << 10));
  }

  if (exp >= -14) {
    uint16_t hexp = (uint16_t)((exp + 15) & 0x1F);
    uint32_t mant_rounded = mant + 0x00001000; // rounding
    uint16_t hmant = (uint16_t)(mant_rounded >> 13);
    if (hmant == 0x400) { // mant overflow
      hexp++;
      hmant = 0;
      if (hexp >= 31) {
        return (uint16_t)(hsign | (0x1F << 10));
      }
    }
    return (uint16_t)(hsign | (hexp << 10) | (hmant & 0x3FF));
  }

  // subnormal or underflow to zero
  if (exp < -24) {
    return hsign;
  }

  // subnormal
  uint32_t mant_full = mant | 0x800000;
  int shift = (-14 - exp);
  uint32_t sub = mant_full >> (shift + 13);
  if ((mant_full >> (shift + 12)) & 1) {
    sub++;
  }
  return (uint16_t)(hsign | (uint16_t)(sub & 0x3FF));
}

static float fp32_from_fp16(uint16_t h) {
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;

  if (exp == 0x1F) { // Inf/NaN
    uint32_t x = (sign << 31) | 0x7F800000 | (mant << 13);
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
  }

  if (exp == 0) { // subnormal or zero
    if (mant == 0) {
      uint32_t x = (sign << 31);
      float f;
      std::memcpy(&f, &x, sizeof(f));
      return f;
    }
    // subnormal
    float f = (sign ? -1.0f : 1.0f) * std::pow(2.0f, -14.0f) *
              ((float)mant / 1024.0f);
    return f;
  }

  uint32_t x = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  float f;
  std::memcpy(&f, &x, sizeof(f));
  return f;
}

// ============================================================================
// Binary gradient packet (little-endian)
// ============================================================================
//
// Header:
//   char magic[4] = "DGRD"
//   u16 version = 1
//   u16 flags = 0
//   u32 step
//   u32 node_id_len
//   u8  node_id[node_id_len]
//   f32 train_loss
//   u32 samples
//   u32 n_tensors
//
// For each tensor update:
//   u32 tensor_id
//   u32 nnz
//   then nnz times:
//     u32 index
//     f32 value
//

struct GradTensorUpdate {
  uint32_t tensor_id = 0;
  bool dense =
      false; // Add flag for dense mode to avoid materializing index 0..N-1
  std::vector<uint32_t> indices;
  std::vector<float> values;
};

struct GradPacket {
  uint32_t step = 0;
  std::string node_id;
  float train_loss = 0.0f;
  uint32_t samples = 0;
  std::vector<GradTensorUpdate> tensors;
  size_t bytes = 0;
};

struct BinaryReader {
  const uint8_t *p = nullptr;
  const uint8_t *end = nullptr;

  explicit BinaryReader(const std::vector<uint8_t> &b)
      : p(b.data()), end(b.data() + b.size()) {}

  bool read_bytes(void *dst, size_t n) {
    if ((size_t)(end - p) < n) {
      return false;
    }
    std::memcpy(dst, p, n);
    p += n;
    return true;
  }

  template <class T> bool read_le(T &out) {
    return read_bytes(&out, sizeof(T));
  }

  bool read_str(std::string &out, size_t n) {
    if ((size_t)(end - p) < n) {
      return false;
    }
    out.assign((const char *)p, (const char *)p + n);
    p += n;
    return true;
  }

  size_t remaining() const { return (size_t)(end - p); }
};

static bool parse_grad_packet(const std::vector<uint8_t> &body, GradPacket &out,
                              ModelParams &model, std::string &err) {
  out = GradPacket{};
  out.bytes = body.size();

  BinaryReader r(body);

  char magic[4];
  uint16_t version = 0;
  uint16_t flags = 0;

  if (!r.read_bytes(magic, 4)) {
    err = "packet too short";
    return false;
  }
  if (!(magic[0] == 'D' && magic[1] == 'G' && magic[2] == 'R' &&
        magic[3] == 'D')) {
    err = "bad magic";
    return false;
  }
  if (!r.read_le(version) || !r.read_le(flags)) {
    err = "bad header";
    return false;
  }
  if (version != 1) {
    err = "unsupported version";
    return false;
  }

  uint32_t step = 0;
  uint32_t node_len = 0;
  float train_loss = 0.0f;
  uint32_t samples = 0;
  uint32_t n_tensors = 0;

  if (!r.read_le(step) || !r.read_le(node_len)) {
    err = "bad step/node_len";
    return false;
  }
  if (node_len > 1024) {
    err = "node_id too long";
    return false;
  }

  std::string node_id;
  if (!r.read_str(node_id, node_len)) {
    err = "bad node_id";
    return false;
  }

  if (!r.read_le(train_loss) || !r.read_le(samples) || !r.read_le(n_tensors)) {
    err = "bad loss/samples/n_tensors";
    return false;
  }

  if (n_tensors > 100000) {
    err = "n_tensors too large";
    return false;
  }

  out.step = step;
  out.node_id = std::move(node_id);
  out.train_loss = train_loss;
  out.samples = samples;
  out.tensors.clear();
  out.tensors.reserve(n_tensors);

  for (uint32_t ti = 0; ti < n_tensors; ti++) {
    uint32_t tensor_id = 0;
    uint32_t nnz = 0;
    if (!r.read_le(tensor_id) || !r.read_le(nnz)) {
      err = "bad tensor header";
      return false;
    }

    GradTensorUpdate u;
    u.tensor_id = tensor_id;

    if (nnz == 0) {
      // Mode 3: Dense FP16
      Tensor *t = model.tensor_by_id(tensor_id);
      if (!t) {
        err = "tensor id not found: " + std::to_string(tensor_id);
        return false;
      }
      int64_t n = t->size;
      u.dense = true;
      u.values.resize(n);
      for (int64_t k = 0; k < n; k++) {
        uint16_t h;
        if (!r.read_le(h)) {
          err = "dense fp16 short";
          return false;
        }
        u.values[k] = fp32_from_fp16(h);
      }
    } else if (flags == 0) {
      // Mode 1: Standard Sparse (u32 index, f32 value)
      if (nnz > 50'000'000) {
        err = "nnz too large";
        return false;
      }
      Tensor *t = model.tensor_by_id(tensor_id);
      if (!t) {
        err = "tensor id not found: " + std::to_string(tensor_id);
        return false;
      }
      u.indices.resize(nnz);
      u.values.resize(nnz);
      for (uint32_t k = 0; k < nnz; k++) {
        uint32_t idx = 0;
        float val = 0.0f;
        if (!r.read_le(idx) || !r.read_le(val)) {
          err = "bad nnz entries (mode 1)";
          return false;
        }
        if (idx >= (uint32_t)t->size) {
          err = "index out of range (mode 1)";
          return false;
        }
        u.indices[k] = idx;
        u.values[k] = val;
      }
    } else if (flags == 1) {
      // Mode 2: Compressed Sparse (u8 skip, fp16 value)
      // NOTE: Mode 2 uses u8 skip. First index limit 0-255, gaps 1-256.
      if (nnz > 50'000'000) {
        err = "nnz too large";
        return false;
      }
      Tensor *t = model.tensor_by_id(tensor_id);
      if (!t) {
        err = "tensor id not found: " + std::to_string(tensor_id);
        return false;
      }
      u.indices.resize(nnz);
      u.values.resize(nnz);
      uint32_t cur_idx = 0;
      for (uint32_t k = 0; k < nnz; k++) {
        uint8_t skip = 0;
        uint16_t h = 0;
        if (!r.read_le(skip) || !r.read_le(h)) {
          err = "bad nnz entries (mode 2)";
          return false;
        }
        if (k == 0) {
          cur_idx = (uint32_t)skip;
        } else {
          cur_idx += (uint32_t)skip + 1;
        }
        if (cur_idx >= (uint32_t)t->size) {
          err = "index out of range (mode 2)";
          return false;
        }
        u.indices[k] = cur_idx;
        u.values[k] = fp32_from_fp16(h);
      }
    } else {
      err = "unsupported flags for version 1";
      return false;
    }

    out.tensors.push_back(std::move(u));
  }

  if (r.remaining() != 0) {
    err = "trailing bytes";
    return false;
  }

  return true;
}

// ============================================================================
// Sparse AdamW (lazy, per-index state)
// ============================================================================

struct SparseAdamElem {
  float m = 0.0f;
  float v = 0.0f;

  void save(std::ostream &os) const {
    os.write((const char *)&m, sizeof(m));
    os.write((const char *)&v, sizeof(v));
    os.write((const char *)&last_step, sizeof(last_step));
  }
  void load(std::istream &is) {
    is.read((char *)&m, sizeof(m));
    is.read((char *)&v, sizeof(v));
    is.read((char *)&last_step, sizeof(last_step));
  }
  uint32_t last_step = 0; // last step when this element was updated
};

struct SparseAdamTensorState {
  std::unordered_map<uint32_t, SparseAdamElem> elems;
};

class SparseAdamW {
public:
  explicit SparseAdamW(TrainConfig tc) : tc_(tc) {}

  void step_tensor(uint32_t global_step, Tensor &param,
                   std::unordered_map<uint32_t, float> &grad_sum,
                   float inv_nodes, SparseAdamTensorState &st) {
    const float lr = tc_.learning_rate;
    const float b1 = tc_.beta1;
    const float b2 = tc_.beta2;
    const float eps = tc_.eps;
    const float wd = tc_.weight_decay;

    // decoupled weight decay multiplier per step
    const float wd_mul = 1.0f - lr * wd;

    for (auto &kv : grad_sum) {
      uint32_t idx = kv.first;
      float g = kv.second * inv_nodes;

      if ((int64_t)idx < 0 || (int64_t)idx >= param.size) {
        continue;
      }

      auto &e = st.elems[idx];
      if (e.last_step == 0) {
        e.last_step = global_step - 1;
      }

      uint32_t delta_steps = global_step - e.last_step;
      if (delta_steps == 0) {
        delta_steps = 1;
      }

      // Lazy weight decay across the gap (includes current step)
      if (wd != 0.0f) {
        float factor = std::pow(wd_mul, (float)delta_steps);
        param.data[idx] *= factor;
      }

      // Lazy moment decay for skipped steps (excluding current)
      uint32_t skipped = (delta_steps >= 1) ? (delta_steps - 1) : 0;
      if (skipped > 0) {
        e.m *= std::pow(b1, (float)skipped);
        e.v *= std::pow(b2, (float)skipped);
      }

      // Adam update for current step
      e.m = b1 * e.m + (1.0f - b1) * g;
      e.v = b2 * e.v + (1.0f - b2) * g * g;

      float m_corr = e.m / (1.0f - std::pow(b1, (float)global_step));
      float v_corr = e.v / (1.0f - std::pow(b2, (float)global_step));

      param.data[idx] -= lr * m_corr / (std::sqrt(v_corr) + eps);
      e.last_step = global_step;
    }
  }

private:
  TrainConfig tc_;
};

// ============================================================================
// Aggregator
// ============================================================================

struct NodeContribution {
  std::string node_id;
  uint32_t step = 0;
  float train_loss = 0.0f;
  uint32_t samples = 0;
  std::vector<GradTensorUpdate> tensors;
  size_t bytes = 0;
};

class Aggregator {
public:
  explicit Aggregator(TrainConfig tc)
      : tc_(tc), opt_(tc), updates_(0), current_step_(1) {}

  uint32_t current_step() const { return current_step_.load(); }
  uint64_t updates() const { return updates_.load(); }

  struct SubmitResult {
    bool ok = false;
    int http_code = 200;
    std::string message;
    uint32_t server_step = 0;
  };

  SubmitResult submit(const GradPacket &p) {
    std::lock_guard<std::mutex> lock(mtx_);

    SubmitResult res;
    res.server_step = current_step_.load();

    if (p.node_id.empty()) {
      res.ok = false;
      res.http_code = 400;
      res.message = "node_id required";
      return res;
    }
    if (p.step + 5 < res.server_step || p.step > res.server_step) {
      res.ok = false;
      res.http_code = 409;
      res.message = "step mismatch (too stale or ahead); fetch latest model";
      return res;
    }
    if (p.samples == 0) {
      res.ok = false;
      res.http_code = 400;
      res.message = "samples must be > 0";
      return res;
    }

    NodeContribution c;
    c.node_id = p.node_id;
    c.step = p.step;
    c.train_loss = p.train_loss;
    c.samples = p.samples;
    c.bytes = p.bytes;
    c.tensors = p.tensors;

    pending_[c.node_id] = std::move(c);

    res.ok = true;
    res.http_code = 200;
    res.message = "ok";
    return res;
  }

  bool ready() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pending_.size() >= (size_t)tc_.min_nodes_for_update;
  }

  // Applies update to model (call with model lock held).
  void aggregate_and_apply(ModelParams &model) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (pending_.empty()) {
      return;
    }

    const uint32_t step = current_step_.load();

    float total_samples = 0.0f;
    float loss_sum = 0.0f;
    for (auto &kv : pending_) {
      auto &c = kv.second;
      total_samples += (float)c.samples;
      loss_sum += c.train_loss * (float)c.samples;
    }
    float avg_loss = (total_samples > 0) ? (loss_sum / total_samples) : 0.0f;
    loss_history_.push_back(avg_loss);

    const int n_nodes = (int)pending_.size();
    const float inv_total_samples =
        (total_samples > 0.0f) ? (1.0f / total_samples) : 0.0f;

    // We used to use a global inv_nodes, but now we weight by samples.
    // The opt_.step_tensor expects (grad_sum * inv_divisor).
    // If we use inv_divisor = 1.0, then grad_sum must be the average gradient.

    // Process each tensor
    auto manifest = model.manifest();
    for (const auto &meta : manifest) {
      uint32_t tensor_id = meta.id;
      Tensor *t = model.tensor_by_id(tensor_id);
      if (!t)
        continue;

      // Accumulate gradients for this tensor
      std::unordered_map<uint32_t, float> grad_avg;
      bool any_contribution = false;

      for (auto &kv : pending_) {
        const auto &c = kv.second;
        for (const auto &tu : c.tensors) {
          if (tu.tensor_id != tensor_id)
            continue;
          any_contribution = true;

          float weight = (float)c.samples * inv_total_samples;
          if (tu.dense) {
            // Optimized dense accumulation
            for (size_t i = 0; i < tu.values.size(); i++) {
              grad_avg[(uint32_t)i] += tu.values[i] * weight;
            }
          } else {
            // Sparse accumulation
            for (size_t i = 0; i < tu.indices.size(); i++) {
              grad_avg[tu.indices[i]] += tu.values[i] * weight;
            }
          }
        }
      }

      if (any_contribution) {
        auto &st = adam_[tensor_id];
        // We pass inv_nodes = 1.0f because grad_avg is already weighted and
        // averaged
        opt_.step_tensor(step, *t, grad_avg, 1.0f, st);
      }
    }

    updates_++;
    current_step_++;

    LOG_INFO("AGG") << "step=" << step << " nodes=" << n_nodes
                    << " avg_loss=" << avg_loss
                    << " updates=" << updates_.load();

    pending_.clear();
  }

  std::vector<float> loss_history() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return loss_history_;
  }

  void save(std::ostream &os) const {
    std::lock_guard<std::mutex> lock(mtx_);
    uint64_t up = updates_.load();
    uint32_t st = current_step_.load();
    os.write((const char *)&up, sizeof(up));
    os.write((const char *)&st, sizeof(st));

    uint64_t lh_size = (uint64_t)loss_history_.size();
    os.write((const char *)&lh_size, sizeof(lh_size));
    if (lh_size > 0) {
      os.write((const char *)loss_history_.data(), lh_size * sizeof(float));
    }

    uint64_t adam_count = (uint64_t)adam_.size();
    os.write((const char *)&adam_count, sizeof(adam_count));
    for (auto &kv : adam_) {
      uint32_t tensor_id = kv.first;
      os.write((const char *)&tensor_id, sizeof(tensor_id));
      auto &ast = kv.second.elems;
      uint64_t elem_count = (uint64_t)ast.size();
      os.write((const char *)&elem_count, sizeof(elem_count));
      for (auto &ekv : ast) {
        uint32_t idx = ekv.first;
        os.write((const char *)&idx, sizeof(idx));
        ekv.second.save(os);
      }
    }
  }

  void load(std::istream &is) {
    std::lock_guard<std::mutex> lock(mtx_);
    uint64_t up = 0;
    uint32_t st = 0;
    is.read((char *)&up, sizeof(up));
    is.read((char *)&st, sizeof(st));
    updates_.store(up);
    current_step_.store(st);

    uint64_t lh_size = 0;
    is.read((char *)&lh_size, sizeof(lh_size));
    loss_history_.resize(lh_size);
    if (lh_size > 0) {
      is.read((char *)loss_history_.data(), lh_size * sizeof(float));
    }

    uint64_t adam_count = 0;
    is.read((char *)&adam_count, sizeof(adam_count));
    adam_.clear();
    for (uint64_t i = 0; i < adam_count; i++) {
      uint32_t tensor_id = 0;
      is.read((char *)&tensor_id, sizeof(tensor_id));
      auto &ast = adam_[tensor_id].elems;
      uint64_t elem_count = 0;
      is.read((char *)&elem_count, sizeof(elem_count));
      for (uint64_t j = 0; j < elem_count; j++) {
        uint32_t idx = 0;
        is.read((char *)&idx, sizeof(idx));
        ast[idx].load(is);
      }
    }
  }

private:
  TrainConfig tc_;
  SparseAdamW opt_;

  mutable std::mutex mtx_;
  std::unordered_map<std::string, NodeContribution> pending_;
  std::unordered_map<uint32_t, SparseAdamTensorState> adam_;

  std::atomic<uint64_t> updates_;
  std::atomic<uint32_t> current_step_;
  std::vector<float> loss_history_;
};

// ============================================================================
// Thread pool
// ============================================================================

class ThreadPool {
public:
  explicit ThreadPool(size_t n) : stop_(false) {
    n = std::max<size_t>(1, n);
    for (size_t i = 0; i < n; i++) {
      workers_.emplace_back([this]() { this->run(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &t : workers_) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

  void enqueue(std::function<void()> fn) {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      q_.push(std::move(fn));
    }
    cv_.notify_one();
  }

private:
  void run() {
    while (true) {
      std::function<void()> fn;
      {
        std::unique_lock<std::mutex> lock(mtx_);
        cv_.wait(lock, [&]() { return stop_ || !q_.empty(); });
        if (stop_ && q_.empty()) {
          return;
        }
        fn = std::move(q_.front());
        q_.pop();
      }
      fn();
    }
  }

  std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<std::function<void()>> q_;
  bool stop_;
  std::vector<std::thread> workers_;
};

// ============================================================================
// Server
// ============================================================================

class Server {
public:
  Server(int port, ModelConfig mc, TrainConfig tc)
      : port_(port), model_(std::move(mc)), tc_(tc), agg_(tc),
        pool_(std::max(2u, std::thread::hardware_concurrency())) {}

  void run() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
      LOG_ERROR("SERVER") << "WSAStartup failed";
      return;
    }
#endif

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif

    listen_sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_sock_ == INVALID_SOCKET) {
      LOG_ERROR("SERVER") << "socket() failed";
      return;
    }

    int opt = 1;
    setsockopt(listen_sock_, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt,
               sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port_);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (::bind(listen_sock_, (sockaddr *)&addr, sizeof(addr)) == SOCKET_ERROR) {
      LOG_ERROR("SERVER") << "bind() failed";
      close_socket(listen_sock_);
      listen_sock_ = INVALID_SOCKET;
      return;
    }

    if (::listen(listen_sock_, 128) == SOCKET_ERROR) {
      LOG_ERROR("SERVER") << "listen() failed";
      close_socket(listen_sock_);
      listen_sock_ = INVALID_SOCKET;
      return;
    }

    if (!load_latest_checkpoint()) {
      LOG_INFO("SERVER") << "Starting fresh training (no checkpoint found)";
    }

    start_update_thread();

    LOG_INFO("SERVER") << "Listening on http://localhost:" << port_;

    while (running_.load()) {
      SOCKET client = ::accept(listen_sock_, nullptr, nullptr);
      if (client == INVALID_SOCKET) {
        if (!running_.load()) {
          break;
        }
        continue;
      }
      pool_.enqueue([this, client]() { this->handle_client(client); });
    }

    stop_update_thread();

    if (listen_sock_ != INVALID_SOCKET) {
      close_socket(listen_sock_);
      listen_sock_ = INVALID_SOCKET;
    }

#ifdef _WIN32
    WSACleanup();
#endif
  }

  void stop() {
    running_.store(false);
    if (listen_sock_ != INVALID_SOCKET) {
#ifdef _WIN32
      ::shutdown(listen_sock_, SD_BOTH);
#else
      ::shutdown(listen_sock_, SHUT_RDWR);
#endif
      close_socket(listen_sock_);
      listen_sock_ = INVALID_SOCKET;
    }
  }

private:
  void start_update_thread() {
    updater_running_.store(true);
    update_thread_ = std::thread([this]() {
      while (updater_running_.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        if (!running_.load()) {
          break;
        }
        if (agg_.ready()) {
          std::unique_lock<std::shared_mutex> lock(model_mtx_);
          agg_.aggregate_and_apply(model_);
          save_checkpoint();
        }
      }
    });
  }

  void stop_update_thread() {
    updater_running_.store(false);
    if (update_thread_.joinable()) {
      update_thread_.join();
    }
  }

  static HttpResponse cors_preflight() {
    HttpResponse r;
    r.status = 200;
    r.status_text = "OK";
    r.content_type = "text/plain; charset=utf-8";
    r.body.clear();
    return r;
  }

  std::optional<HttpResponse> try_serve_static(const HttpRequest &req) {
    if (req.method != "GET") {
      return std::nullopt;
    }

    // Do not intercept API routes.
    if (req.path.rfind("/api/", 0) == 0) {
      return std::nullopt;
    }
    if (req.path == "/healthz") {
      return std::nullopt;
    }

    std::string rel = req.path;
    if (rel == "/") {
      rel = "/index.html";
    }
    if (!rel.empty() && rel[0] == '/') {
      rel.erase(rel.begin());
    }
    while (!rel.empty() && rel[0] == '/') {
      rel.erase(rel.begin());
    }

    if (rel.empty()) {
      return HttpResponse::text("Not Found", 404);
    }
    if (!is_safe_rel_path(rel)) {
      return HttpResponse::text("Bad path", 400);
    }

    std::filesystem::path full =
        (std::filesystem::path(static_dir_) / std::filesystem::path(rel))
            .lexically_normal();

    // Ensure the resolved path still sits inside static_dir_ lexically.
    std::filesystem::path base =
        std::filesystem::path(static_dir_).lexically_normal();
    std::string full_s = full.generic_string();
    std::string base_s = base.generic_string();
    if (full_s.rfind(base_s, 0) != 0) {
      return HttpResponse::text("Bad path", 400);
    }

    std::ifstream f(full, std::ios::binary);
    if (!f.is_open()) {
      return HttpResponse::text("Not Found", 404);
    }

    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    HttpResponse res =
        HttpResponse::bin(std::move(bytes), mime_type_from_path(rel), 200);

    // Light caching policy: no-cache for html, short cache for assets.
    if (rel.size() >= 5 && rel.substr(rel.size() - 5) == ".html") {
      res.headers.push_back({"Cache-Control", "no-cache"});
    } else {
      res.headers.push_back({"Cache-Control", "public, max-age=60"});
    }

    return res;
  }

  void save_checkpoint() {
    uint32_t step = agg_.current_step();
    std::error_code ec;
    if (!std::filesystem::exists("checkpoints", ec)) {
      std::filesystem::create_directory("checkpoints", ec);
    }

    std::string filename =
        "checkpoints/checkpoint_step_" + std::to_string(step) + ".bin";
    std::ofstream os(filename, std::ios::binary);
    if (!os.is_open()) {
      LOG_ERROR("SERVER") << "Failed to open checkpoint for writing: "
                          << filename;
      return;
    }

    agg_.save(os);
    model_.save(os);
    os.close();

    LOG_INFO("SERVER") << "Checkpoint saved: " << filename;

    // Rolling logic: keep last 5
    std::vector<std::pair<uint32_t, std::filesystem::path>> paths;
    for (auto &p : std::filesystem::directory_iterator("checkpoints")) {
      std::string name = p.path().filename().string();
      if (p.path().extension() == ".bin" &&
          name.find("checkpoint_step_") == 0) {
        try {
          uint32_t s = std::stoul(name.substr(16));
          paths.push_back({s, p.path()});
        } catch (...) {
        }
      }
    }
    if (paths.size() > 5) {
      std::sort(paths.begin(), paths.end());
      for (size_t i = 0; i < paths.size() - 5; i++) {
        std::filesystem::remove(paths[i].second, ec);
      }
    }
  }

  bool load_latest_checkpoint() {
    std::error_code ec;
    if (!std::filesystem::exists("checkpoints", ec)) {
      return false;
    }

    std::vector<std::pair<uint32_t, std::filesystem::path>> checkpoints;
    for (auto &p : std::filesystem::directory_iterator("checkpoints")) {
      std::string name = p.path().filename().string();
      if (p.path().extension() == ".bin" &&
          name.find("checkpoint_step_") == 0) {
        try {
          uint32_t step =
              std::stoul(name.substr(16)); // "checkpoint_step_".length()
          checkpoints.push_back({step, p.path()});
        } catch (...) {
        }
      }
    }

    if (checkpoints.empty()) {
      return false;
    }

    std::sort(checkpoints.begin(), checkpoints.end());
    auto &latest = checkpoints.back();

    std::ifstream is(latest.second, std::ios::binary);
    if (!is.is_open()) {
      LOG_ERROR("SERVER") << "Failed to open checkpoint for loading: "
                          << latest.second;
      return false;
    }

    agg_.load(is);
    model_.load(is);
    LOG_INFO("SERVER") << "Resumed from checkpoint: " << latest.second;
    return true;
  }

  static std::string json_escape(const std::string &s) {
    std::ostringstream o;
    for (char c : s) {
      switch (c) {
      case '\\':
        o << "\\\\";
        break;
      case '"':
        o << "\\\"";
        break;
      case '\n':
        o << "\\n";
        break;
      case '\r':
        o << "\\r";
        break;
      case '\t':
        o << "\\t";
        break;
      default:
        if ((unsigned char)c < 0x20) {
          o << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << (int)(unsigned char)c << std::dec;
        } else {
          o << c;
        }
      }
    }
    return o.str();
  }

  HttpResponse route(const HttpRequest &req) {
    if (req.method == "OPTIONS") {
      return cors_preflight();
    }

    // if (req.method == "GET" && req.path == "/") {
    //   return HttpResponse::text(
    //       "ok\n"
    //       "GET  /api/v1/model/info\n"
    //       "GET  /api/v1/model/manifest\n"
    //       "GET  /api/v1/model/tensor/{id}?format=f16|f32&offset=&count=\n"
    //       "POST /api/v1/train/submit (binary DGRD)\n"
    //       "GET  /api/v1/server/losses\n");
    // }
    // Static files (index.html, app.js, style.css, etc)
    auto sres = try_serve_static(req);
    if (sres) {
      return *sres;
    }

    if (req.method == "GET" && req.path == "/healthz") {
      return HttpResponse::json("{\"ok\":true}");
    }

    if (req.method == "GET" && req.path == "/api/v1/model/info") {
      std::shared_lock<std::shared_mutex> lock(model_mtx_);
      std::ostringstream ss;
      ss << "{"
         << "\"step\":" << agg_.current_step() << ","
         << "\"updates\":" << agg_.updates() << ","
         << "\"total_params\":" << model_.total_params() << ","
         << "\"config\":{"
         << "\"vocab_size\":" << model_.cfg.vocab_size << ","
         << "\"d_model\":" << model_.cfg.d_model << ","
         << "\"n_heads\":" << model_.cfg.n_heads << ","
         << "\"n_layers\":" << model_.cfg.n_layers << ","
         << "\"d_ff\":" << model_.cfg.d_ff << ","
         << "\"max_seq_len\":" << model_.cfg.max_seq_len << "},"
         << "\"train\":{"
         << "\"learning_rate\":" << tc_.learning_rate << ","
         << "\"beta1\":" << tc_.beta1 << ","
         << "\"beta2\":" << tc_.beta2 << ","
         << "\"eps\":" << tc_.eps << ","
         << "\"weight_decay\":" << tc_.weight_decay << ","
         << "\"min_nodes_for_update\":" << tc_.min_nodes_for_update << "}"
         << "}";
      return HttpResponse::json(ss.str());
    }

    if (req.method == "GET" && req.path == "/api/v1/model/manifest") {
      std::shared_lock<std::shared_mutex> lock(model_mtx_);
      auto man = model_.manifest();
      std::ostringstream ss;
      ss << "{"
         << "\"step\":" << agg_.current_step() << ","
         << "\"tensors\":[";
      for (size_t i = 0; i < man.size(); i++) {
        const auto &t = man[i];
        ss << "{"
           << "\"id\":" << t.id << ","
           << "\"name\":\"" << json_escape(t.name) << "\","
           << "\"shape\":[";
        for (size_t j = 0; j < t.shape.size(); j++) {
          ss << t.shape[j] << (j + 1 == t.shape.size() ? "" : ",");
        }
        ss << "],"
           << "\"elements\":" << t.elements << ","
           << "\"bytes_f32\":" << (t.elements * 4) << ","
           << "\"bytes_f16\":" << (t.elements * 2) << "}";
        ss << (i + 1 == man.size() ? "" : ",");
      }
      ss << "]}";
      return HttpResponse::json(ss.str());
    }

    // GET /api/v1/model/tensor/{id}?format=f16|f32&offset=0&count=...
    if (req.method == "GET" &&
        req.path.rfind("/api/v1/model/tensor/", 0) == 0) {
      std::string sid =
          req.path.substr(std::string("/api/v1/model/tensor/").size());
      if (sid.empty()) {
        return HttpResponse::text("missing tensor id", 400);
      }

      uint32_t tid = 0;
      try {
        tid = (uint32_t)std::stoul(sid);
      } catch (...) {
        return HttpResponse::text("bad tensor id", 400);
      }

      auto q = parse_query_kv(req.query);
      std::string format = "f16";
      if (q.count("format")) {
        format = q["format"];
      }
      if (format != "f16" && format != "f32") {
        return HttpResponse::text("format must be f16 or f32", 400);
      }

      int64_t offset = 0;
      int64_t count = -1;
      if (q.count("offset")) {
        try {
          offset = (int64_t)std::stoll(q["offset"]);
        } catch (...) {
          return HttpResponse::text("bad offset", 400);
        }
      }
      if (q.count("count")) {
        try {
          count = (int64_t)std::stoll(q["count"]);
        } catch (...) {
          return HttpResponse::text("bad count", 400);
        }
      }

      std::shared_lock<std::shared_mutex> lock(model_mtx_);
      Tensor *t = model_.tensor_by_id(tid);
      if (!t) {
        return HttpResponse::text("tensor not found", 404);
      }

      int64_t n = t->size;
      if (offset < 0 || offset > n) {
        return HttpResponse::text("offset out of range", 416);
      }

      int64_t max_count = n - offset;
      if (count < 0) {
        count = max_count;
      }
      if (count < 0 || count > max_count) {
        return HttpResponse::text("count out of range", 416);
      }

      std::vector<uint8_t> payload;
      payload.reserve((size_t)count * (format == "f16" ? 2 : 4));

      if (format == "f32") {
        payload.resize((size_t)count * 4);
        std::memcpy(payload.data(), t->data.data() + offset, (size_t)count * 4);
      } else {
        payload.resize((size_t)count * 2);
        uint16_t *dst = (uint16_t *)payload.data();
        for (int64_t i = 0; i < count; i++) {
          dst[i] = fp16_from_fp32(t->data[(size_t)(offset + i)]);
        }
      }

      HttpResponse res =
          HttpResponse::bin(std::move(payload), "application/octet-stream");
      res.headers.push_back(
          {"X-Model-Step", std::to_string(agg_.current_step())});
      res.headers.push_back({"X-Tensor-Id", std::to_string(tid)});
      res.headers.push_back({"X-Tensor-Offset", std::to_string(offset)});
      res.headers.push_back({"X-Tensor-Count", std::to_string(count)});
      res.headers.push_back({"X-Tensor-Format", format});
      return res;
    }

    // POST /api/v1/train/submit (binary DGRD)
    if (req.method == "POST" && req.path == "/api/v1/train/submit") {
      auto ct_it = req.headers.find("content-type");
      if (ct_it == req.headers.end()) {
        return HttpResponse::text("Content-Type required", 400);
      }
      std::string ct = to_lower(ct_it->second);
      if (ct.find("application/octet-stream") == std::string::npos) {
        return HttpResponse::text(
            "Content-Type must be application/octet-stream", 415);
      }

      GradPacket p;
      std::string perr;
      if (!parse_grad_packet(req.body, p, model_, perr)) {
        return HttpResponse::json(std::string("{\"ok\":false,\"error\":\"") +
                                      json_escape(perr) + "\"}",
                                  400);
      }

      auto res = agg_.submit(p);
      std::ostringstream ss;
      ss << "{"
         << "\"ok\":" << (res.ok ? "true" : "false") << ","
         << "\"message\":\"" << json_escape(res.message) << "\","
         << "\"server_step\":" << res.server_step << "}";
      return HttpResponse::json(ss.str(), res.http_code);
    }

    // GET /api/v1/server/losses
    if (req.method == "GET" && req.path == "/api/v1/server/losses") {
      auto losses = agg_.loss_history();
      std::ostringstream ss;
      ss << "[";
      for (size_t i = 0; i < losses.size(); i++) {
        ss << losses[i] << (i + 1 == losses.size() ? "" : ",");
      }
      ss << "]";
      return HttpResponse::json(ss.str());
    }

    return HttpResponse::text("Not Found", 404);
  }

  void handle_client(SOCKET s) {
    HttpRequest req;
    std::string err;

    bool ok = read_http_request(s, req, err);
    if (!ok) {
      HttpResponse res = HttpResponse::text("Bad Request: " + err, 400);
      std::string hdr = http_serialize_headers(res);
      send_all(s, hdr);
      send_all(s, res.body.data(), res.body.size());
      close_socket(s);
      return;
    }

    HttpResponse res;
    try {
      res = route(req);
    } catch (const std::exception &e) {
      res = HttpResponse::text(std::string("Internal error: ") + e.what(), 500);
    } catch (...) {
      res = HttpResponse::text("Internal error", 500);
    }

    std::string hdr = http_serialize_headers(res);
    send_all(s, hdr);
    if (!res.body.empty()) {
      send_all(s, res.body.data(), res.body.size());
    }

    close_socket(s);
  }

private:
  int port_;
  ModelParams model_;
  TrainConfig tc_;
  Aggregator agg_;

  std::shared_mutex model_mtx_;
  std::string static_dir_ = "static";

  std::atomic<bool> running_{true};
  SOCKET listen_sock_{INVALID_SOCKET};

  std::atomic<bool> updater_running_{false};
  std::thread update_thread_;

  ThreadPool pool_;
};

// ============================================================================
// main
// ============================================================================

static Server *g_server = nullptr;

#ifndef _WIN32
static void on_sigint(int) {
  if (g_server) {
    g_server->stop();
  }
}
#endif

int main(int argc, char **argv) {
  int port = 8080;
  bool tiny = false;

  if (argc >= 2) {
    std::string a1 = argv[1];
    if (a1 != "--tiny") {
      port = std::atoi(argv[1]);
    }
  }
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--tiny") {
      tiny = true;
    }
  }

  ModelConfig mc;
  TrainConfig tc;

  if (tiny) {
    // Much smaller model so the server is easy to run on a laptop.
    mc.vocab_size = 8192;
    mc.d_model = 256;
    mc.n_heads = 8;
    mc.n_layers = 4;
    mc.d_ff = 1024;
    mc.max_seq_len = 256;

    tc.learning_rate = 5e-4f;
    tc.min_nodes_for_update = 1;
  }

  // Override with JSON if exists
  std::string mj = read_file_to_string("static/model_config.json");
  if (!mj.empty()) {
    mc.load_json(mj);
    LOG_INFO("SERVER") << "Loaded model_config.json";
  }
  std::string tj = read_file_to_string("static/train_config.json");
  if (!tj.empty()) {
    tc.load_json(tj);
    LOG_INFO("SERVER") << "Loaded train_config.json";
  }

#ifndef _WIN32
  signal(SIGINT, on_sigint);
#endif

  try {
    Server server(port, mc, tc);
    g_server = &server;
    server.run();
    g_server = nullptr;
  } catch (const std::exception &e) {
    LOG_ERROR("FATAL") << e.what();
    return 1;
  } catch (...) {
    LOG_ERROR("FATAL") << "unknown error";
    return 1;
  }

  return 0;
}