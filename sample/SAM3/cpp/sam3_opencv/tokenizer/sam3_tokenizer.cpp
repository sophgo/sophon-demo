//===----------------------------------------------------------------------===//
//
// Copyright (C) 2022 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#include "sam3_tokenizer.hpp"

#include <miniz.h>

#include <algorithm>
#include <cstring>

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

// CLIP vocabulary specials.
constexpr int32_t kSot = 49406;  // <start_of_text>
constexpr int32_t kEot = 49407;  // <end_of_text>
// 256 base entries + 256 "</w>" variants precede the merge results.
constexpr int32_t kNumMerges = 48894;

std::string encodeUtf8(uint32_t cp) {
  std::string s;
  if (cp < 0x80) {
    s.push_back((char)cp);
  } else if (cp < 0x800) {
    s.push_back((char)(0xC0 | (cp >> 6)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  } else {
    s.push_back((char)(0xE0 | (cp >> 12)));
    s.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back((char)(0x80 | (cp & 0x3F)));
  }
  return s;
}

}  // namespace

Sam3Tokenizer::Sam3Tokenizer(const std::string& bpe_gz_path,
                             int context_length)
    : context_length_(context_length), bpe_path_(bpe_gz_path) {
  buildByteEncoder();

  const std::string raw = gunzip(bpe_gz_path);

  // File layout: line 0 is the version header, then 48894 "first second"
  // merge pairs.
  std::vector<std::string> lines;
  {
    std::istringstream ss(raw);
    std::string line;
    while (std::getline(ss, line)) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
    }
  }
  if (lines.size() < (size_t)kNumMerges + 1) {
    throw std::runtime_error("BPE vocab too small: " +
                             std::to_string(lines.size()) + " lines");
  }

  for (int32_t m = 0; m < kNumMerges; ++m) {
    const std::string& line = lines[m + 1];
    const size_t sp = line.find(' ');
    if (sp == std::string::npos) {
      throw std::runtime_error("bad BPE merges line " + std::to_string(m + 1));
    }
    const std::string first = line.substr(0, sp);
    const std::string second = line.substr(sp + 1);
    bpe_ranks_[{first, second}] = m;
    encoder_[first + second] = 512 + m;
  }
}

void Sam3Tokenizer::buildByteEncoder() {
  // bytes_to_unicode(): printable ASCII and part of latin-1 map to
  // themselves, the remaining bytes map to 256+n.
  std::vector<uint32_t> bs;
  for (uint32_t b = '!'; b <= '~'; ++b) bs.push_back(b);
  for (uint32_t b = 161; b <= 172; ++b) bs.push_back(b);
  for (uint32_t b = 174; b <= 255; ++b) bs.push_back(b);
  std::vector<bool> in_bs(256, false);
  for (uint32_t b : bs) in_bs[b] = true;

  byte_encoder_.resize(256);
  uint32_t n = 0;
  for (int b = 0; b < 256; ++b) {
    const uint32_t u = in_bs[b] ? (uint32_t)b : 256 + n++;
    byte_encoder_[b] = encodeUtf8(u);
  }

  // Base vocab: byte chars in bs order (ids 0..255), then the same chars
  // with an appended "</w>" (ids 256..511). Merge results get 512+rank in
  // the constructor, matching the python construction order.
  for (size_t i = 0; i < bs.size(); ++i) {
    const std::string ch = encodeUtf8(bs[i]);
    encoder_[ch] = (int32_t)i;
    encoder_[ch + "</w>"] = (int32_t)(256 + i);
  }
}

std::string Sam3Tokenizer::clean(const std::string& text) {
  // whitespace_clean(): collapse whitespace runs to single spaces, strip,
  // then lower() (ASCII only).
  std::string out;
  bool in_ws = false;
  for (char c : text) {
    const bool ws = (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                     c == '\f' || c == '\v');
    if (ws) {
      if (!out.empty() && !in_ws) out.push_back(' ');
      in_ws = true;
    } else {
      out.push_back((c >= 'A' && c <= 'Z') ? (char)(c - 'A' + 'a') : c);
      in_ws = false;
    }
  }
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

std::vector<std::string> Sam3Tokenizer::scan(const std::string& text) const {
  // Simplified CLIP regex, tried in this order at each position:
  //   contractions ('s 't 're 've 'm 'll 'd) | [letters]+ | [digit] |
  //   [others]+ (greedy, python re.IGNORECASE)
  // Bytes >= 0x80 are treated as letters.
  std::vector<std::string> tokens;
  size_t i = 0;
  auto isLetter = [](unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
  };
  static const char* kContractions[] = {"s", "t", "re", "ve", "m", "ll", "d"};
  while (i < text.size()) {
    // Whitespace never forms a token (python regex branches all exclude \s).
    {
      const unsigned char w = (unsigned char)text[i];
      if (w == ' ' || w == '\t' || w == '\n' || w == '\r' || w == '\f' ||
          w == '\v') {
        ++i;
        continue;
      }
    }
    if (text[i] == '\'') {
      bool matched = false;
      for (const char* p : kContractions) {
        const size_t plen = strlen(p);
        if (i + 1 + plen > text.size()) continue;
        bool ok = true;
        for (size_t k = 0; k < plen; ++k) {
          char a = text[i + 1 + k];
          if (a >= 'A' && a <= 'Z') a = (char)(a - 'A' + 'a');
          if (a != p[k]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          tokens.emplace_back(text.substr(i, 1 + plen));
          i += 1 + plen;
          matched = true;
          break;
        }
      }
      if (matched) continue;
    }
    const unsigned char c = (unsigned char)text[i];
    if (isLetter(c)) {
      size_t j = i + 1;
      while (j < text.size() && isLetter((unsigned char)text[j])) ++j;
      tokens.emplace_back(text.substr(i, j - i));
      i = j;
    } else if (c >= '0' && c <= '9') {
      tokens.emplace_back(text.substr(i, 1));
      ++i;
    } else {
      // "Other" run: greedy, includes apostrophes that did not start a
      // contraction, stops at whitespace / letters / digits.
      size_t j = i + 1;
      while (j < text.size()) {
        const unsigned char d = (unsigned char)text[j];
        if (d == ' ' || d == '\t' || d == '\n' || d == '\r' || d == '\f' ||
            d == '\v' || isLetter(d) || (d >= '0' && d <= '9')) {
          break;
        }
        ++j;
      }
      tokens.emplace_back(text.substr(i, j - i));
      i = j;
    }
  }
  return tokens;
}

std::vector<std::string> Sam3Tokenizer::bpe(const std::string& token) const {
  auto it = cache_.find(token);
  if (it != cache_.end()) return it->second;

  // Each raw byte maps to one unicode char; the last symbol is end-of-word.
  std::vector<std::string> symbols;
  symbols.reserve(token.size());
  for (size_t k = 0; k < token.size(); ++k) {
    symbols.push_back(byte_encoder_[(unsigned char)token[k]]);
  }
  if (!symbols.empty()) symbols.back() += "</w>";

  while (symbols.size() > 1) {
    // Find the lowest-rank adjacent pair.
    int32_t best_rank = INT32_MAX;
    size_t best_k = 0;
    for (size_t k = 0; k + 1 < symbols.size(); ++k) {
      auto rit = bpe_ranks_.find({symbols[k], symbols[k + 1]});
      if (rit != bpe_ranks_.end() && rit->second < best_rank) {
        best_rank = rit->second;
        best_k = k;
      }
    }
    if (best_rank == INT32_MAX) break;

    // Merge every occurrence of that pair, greedy left to right.
    const std::string first = symbols[best_k];
    const std::string second = symbols[best_k + 1];
    std::vector<std::string> next;
    next.reserve(symbols.size() - 1);
    for (size_t k = 0; k < symbols.size();) {
      if (k + 1 < symbols.size() && symbols[k] == first &&
          symbols[k + 1] == second) {
        next.push_back(first + second);
        k += 2;
      } else {
        next.push_back(symbols[k]);
        ++k;
      }
    }
    symbols = std::move(next);
  }

  cache_[token] = symbols;
  return symbols;
}

std::vector<int32_t> Sam3Tokenizer::encode(const std::string& text) const {
  std::vector<int32_t> ids;
  ids.push_back(kSot);

  for (const std::string& tok : scan(clean(text))) {
    for (const std::string& sym : bpe(tok)) {
      // Symbols carry their own "</w>" marker; vocab ids 256..511 cover the
      // end-of-word variants and 512+ the merge results.
      auto eit = encoder_.find(sym);
      if (eit == encoder_.end()) {
        throw std::runtime_error("token not in SAM3 vocab: " + sym);
      }
      ids.push_back(eit->second);
    }
  }

  ids.push_back(kEot);
  if ((int)ids.size() > context_length_) {
    ids.resize(context_length_);
    ids.back() = kEot;
  }
  ids.resize(context_length_, 0);
  return ids;
}

std::string Sam3Tokenizer::gunzip(const std::string& gz_path) {
  std::ifstream f(gz_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + gz_path);
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string gz = ss.str();

  if (gz.size() < 18 || (unsigned char)gz[0] != 0x1F || (unsigned char)gz[1] != 0x8B) {
    throw std::runtime_error(gz_path + " is not gzip");
  }
  // Trailer: CRC32 + ISIZE (uncompressed size mod 2^32), little-endian.
  uint32_t isize = (unsigned char)gz[gz.size() - 4] |
                   ((unsigned char)gz[gz.size() - 3] << 8) |
                   ((unsigned char)gz[gz.size() - 2] << 16) |
                   ((unsigned char)gz[gz.size() - 1] << 24);

  std::string out(isize, '\0');
  // Skip the 10-byte gzip header; deflate data ends 8 bytes before EOF
  // (this file has no FEXTRA/FNAME/FCOMMENT fields).
  size_t consumed = tinfl_decompress_mem_to_mem(&out[0], out.size(),
                                                gz.data() + 10,
                                                gz.size() - 18, 0);
  if (consumed == TINFL_DECOMPRESS_MEM_TO_MEM_FAILED) {
    throw std::runtime_error("gzip decompress failed for " + gz_path);
  }
  out.resize(consumed);
  return out;
}
