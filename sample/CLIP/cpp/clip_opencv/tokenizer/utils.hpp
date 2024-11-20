#pragma once
#include <iostream>
#include <sstream>
#include <string>

#include <assert.h>
#include <inttypes.h>
#include <stdarg.h>
#include <algorithm>
#include <codecvt>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include "vocab.hpp"

std::string load_merges() {
    std::string merges_utf8_str(reinterpret_cast<const char*>(merges_utf8_c_str),
                                sizeof(merges_utf8_c_str));
    return merges_utf8_str;
}

std::u32string unicode_value_to_utf32(int unicode_value) {
    std::u32string utf32_string = {static_cast<char32_t>(unicode_value)};
    return utf32_string;
}

std::string ltrim(const std::string& s) {
    auto it = std::find_if(s.begin(), s.end(),
                           [](int ch) { return !std::isspace(ch); });
    return std::string(it, s.end());
}

std::string rtrim(const std::string& s) {
    auto it = std::find_if(s.rbegin(), s.rend(),
                           [](int ch) { return !std::isspace(ch); });
    return std::string(s.begin(), it.base());
}

std::string trim(const std::string& s) {
    return rtrim(ltrim(s));
}

std::u32string utf8_to_utf32(const std::string& utf8_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.from_bytes(utf8_str);
}

std::string utf32_to_utf8(const std::u32string& utf32_str) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    return converter.to_bytes(utf32_str);
}

std::vector<std::pair<int, std::u32string>> bytes_to_unicode() {
    std::vector<std::pair<int, std::u32string>> byte_unicode_pairs;
    std::set<int> byte_set;
    for (int b = static_cast<int>('!'); b <= static_cast<int>('~'); ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 161; b <= 172; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    for (int b = 174; b <= 255; ++b) {
        byte_set.insert(b);
        byte_unicode_pairs.push_back(
            std::pair<int, std::u32string>(b, unicode_value_to_utf32(b)));
    }
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (byte_set.find(b) == byte_set.end()) {
            byte_unicode_pairs.push_back(
                std::pair<int, std::u32string>(b, unicode_value_to_utf32(n + 256)));
            ++n;
        }
    }
    return byte_unicode_pairs;
}

static bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() &&
           0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}