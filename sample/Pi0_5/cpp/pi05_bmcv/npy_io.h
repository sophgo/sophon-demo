//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Sophgo Technologies Inc.  All rights reserved.
//
// SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
#ifndef PI05_NPY_IO_H
#define PI05_NPY_IO_H

// Minimal .npy / .npz reader and writer, limited to little-endian float32, which is all the
// pi0.5 assets use. Written in-tree rather than pulling a dependency, to keep the SoC build
// free of extra packages.
//
// CAUTION: .npz files must be written in STORED (uncompressed) mode, i.e. with np.savez and
// not np.savez_compressed. The reader rejects compressed entries explicitly. Historically a
// compressed archive produced silent garbage: either a load failure or an all-zero model
// output, with no exception raised anywhere.

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace pi05 {

// Parsed .npy header: element dtype string, dimensions, and where the payload starts.
struct NpyHeader {
    std::string descr;        // e.g. "<f4", "|u1"
    std::vector<size_t> dims;
    size_t data_off = 0;
    size_t count = 0;         // product of dims
};

// Parses the header of an in-memory .npy buffer, without touching the payload.
//
// in:  data  buffer holding the whole .npy file
//      len   buffer size in bytes
// out: hdr   decoded header
//      return true on success
inline bool npy_parse_header(const uint8_t* data, size_t len, NpyHeader* hdr) {
    static const char kMagic[] = "\x93NUMPY";
    if (len < 10 || memcmp(data, kMagic, 6) != 0) return false;

    const int major = data[6];
    size_t header_len = 0, offset = 0;
    if (major == 1) {
        uint16_t hl = 0;
        memcpy(&hl, data + 8, 2);
        header_len = hl;
        offset = 10;
    } else {
        uint32_t hl = 0;
        memcpy(&hl, data + 8, 4);
        header_len = hl;
        offset = 12;
    }
    if (offset + header_len > len) return false;
    const std::string header(reinterpret_cast<const char*>(data + offset), header_len);

    const size_t dpos = header.find("'descr'");
    if (dpos == std::string::npos) return false;
    const size_t q1 = header.find('\'', header.find(':', dpos) + 1);
    const size_t q2 = header.find('\'', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) return false;
    hdr->descr = header.substr(q1 + 1, q2 - q1 - 1);
    if (header.find("'fortran_order': True") != std::string::npos) return false;

    const size_t lpos = header.find("'shape'");
    const size_t p1 = header.find('(', lpos);
    const size_t p2 = header.find(')', p1);
    if (p1 == std::string::npos || p2 == std::string::npos) return false;
    {
        const std::string s = header.substr(p1 + 1, p2 - p1 - 1);
        size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && !isdigit(static_cast<unsigned char>(s[i]))) i++;
            if (i >= s.size()) break;
            size_t v = 0;
            while (i < s.size() && isdigit(static_cast<unsigned char>(s[i]))) {
                v = v * 10 + static_cast<size_t>(s[i] - '0');
                i++;
            }
            hdr->dims.push_back(v);
        }
    }
    hdr->count = 1;
    for (size_t v : hdr->dims) hdr->count *= v;
    hdr->data_off = offset + header_len;
    return true;
}

// Parses one in-memory little-endian float32 .npy buffer.
//
// in:  data   buffer holding the whole .npy file
//      len    buffer size in bytes
// out: out    decoded float32 values
//      shape  decoded dimensions, may be nullptr
//      return true on success
inline bool npy_parse_f32(const uint8_t* data, size_t len, std::vector<float>* out,
                          std::vector<size_t>* shape) {
    NpyHeader hdr;
    if (!npy_parse_header(data, len, &hdr)) return false;
    if (hdr.descr != "<f4" && hdr.descr != "|f4") return false;
    if (hdr.data_off + hdr.count * 4 > len) return false;
    out->resize(hdr.count);
    if (hdr.count) memcpy(out->data(), data + hdr.data_off, hdr.count * 4);
    if (shape) *shape = hdr.dims;
    return true;
}

// Parses one in-memory uint8 .npy buffer, used for the observation images.
//
// in:  data   buffer holding the whole .npy file
//      len    buffer size in bytes
// out: out    decoded bytes
//      shape  decoded dimensions, may be nullptr
//      return true on success
inline bool npy_parse_u8(const uint8_t* data, size_t len, std::vector<uint8_t>* out,
                         std::vector<size_t>* shape) {
    NpyHeader hdr;
    if (!npy_parse_header(data, len, &hdr)) return false;
    if (hdr.descr != "|u1" && hdr.descr != "<u1") return false;
    if (hdr.data_off + hdr.count > len) return false;
    out->assign(data + hdr.data_off, data + hdr.data_off + hdr.count);
    if (shape) *shape = hdr.dims;
    return true;
}

// Reads a whole file into memory.
//
// in:  path  file path
// out: buf   file contents
//      return true on success
inline bool read_file(const std::string& path, std::vector<uint8_t>* buf) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) {
        fclose(f);
        return false;
    }
    buf->resize(static_cast<size_t>(sz));
    const size_t rd = fread(buf->data(), 1, buf->size(), f);
    fclose(f);
    return rd == buf->size();
}

// Loads a float32 .npy file from disk.
//
// in:  path   file path
// out: out    decoded float32 values
//      shape  decoded dimensions, may be nullptr
//      return true on success
inline bool npy_load_f32(const std::string& path, std::vector<float>* out,
                         std::vector<size_t>* shape = nullptr) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;
    return npy_parse_f32(buf.data(), buf.size(), out, shape);
}

// Loads a uint8 .npy file from disk.
//
// in:  path   file path
// out: out    decoded bytes
//      shape  decoded dimensions, may be nullptr
//      return true on success
inline bool npy_load_u8(const std::string& path, std::vector<uint8_t>* out,
                        std::vector<size_t>* shape = nullptr) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;
    return npy_parse_u8(buf.data(), buf.size(), out, shape);
}

// Reads a little-endian uint16 from a raw buffer.
//
// in:  p  pointer to at least 2 bytes
// out: decoded value
inline uint16_t RdU16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }

// Reads a little-endian uint32 from a raw buffer.
//
// in:  p  pointer to at least 4 bytes
// out: decoded value
inline uint32_t RdU32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// Extracts one entry's raw bytes from an in-memory ZIP archive.
// Entries stored with any compression method other than STORED are rejected.
//
// in:  zip   whole archive
//      name  entry name to look up
// out: out   raw entry bytes
//      return true on success
inline bool ZipExtractStored(const std::vector<uint8_t>& zip, const std::string& name,
                             std::vector<uint8_t>* out) {
    // Locate the end-of-central-directory record by scanning backwards; the trailing comment
    // may be up to 64 KB long.
    const size_t n = zip.size();
    if (n < 22) return false;
    const size_t scan_from = n > (22 + 65535) ? n - (22 + 65535) : 0;
    size_t eocd = std::string::npos;
    for (size_t i = n - 22 + 1; i > scan_from;) {
        --i;
        if (RdU32(&zip[i]) == 0x06054b50u) {
            eocd = i;
            break;
        }
    }
    if (eocd == std::string::npos || eocd + 22 > n) return false;

    const uint16_t total = RdU16(&zip[eocd + 10]);
    const uint32_t cd_off = RdU32(&zip[eocd + 16]);
    if (cd_off > n) return false;

    size_t p = cd_off;
    for (uint16_t i = 0; i < total && p + 46 <= n; i++) {
        if (RdU32(&zip[p]) != 0x02014b50u) return false;
        const uint16_t method = RdU16(&zip[p + 10]);
        const uint32_t comp_sz = RdU32(&zip[p + 20]);
        const uint16_t name_len = RdU16(&zip[p + 28]);
        const uint16_t extra_len = RdU16(&zip[p + 30]);
        const uint16_t cmt_len = RdU16(&zip[p + 32]);
        const uint32_t lho = RdU32(&zip[p + 42]);
        if (p + 46 + name_len > n) return false;
        const std::string entry(reinterpret_cast<const char*>(&zip[p + 46]), name_len);

        if (entry == name) {
            if (method != 0) return false;   // see the header comment: STORED only
            if (lho + 30 > n) return false;
            if (RdU32(&zip[lho]) != 0x04034b50u) return false;
            const uint16_t l_name = RdU16(&zip[lho + 26]);
            const uint16_t l_extra = RdU16(&zip[lho + 28]);
            const size_t data_off = static_cast<size_t>(lho) + 30 + l_name + l_extra;
            if (data_off + comp_sz > n) return false;
            out->assign(zip.begin() + data_off, zip.begin() + data_off + comp_sz);
            return true;
        }
        p += 46 + name_len + extra_len + cmt_len;
    }
    return false;
}

// Loads one float32 array out of a .npz archive. The key names the array, so key "mean" reads
// the member "mean.npy".
//
// in:  path  .npz file path
//      key   array name inside the archive
// out: out   decoded float32 values
//      shape decoded dimensions, may be nullptr
//      return true on success
inline bool npz_load_f32(const std::string& path, const std::string& key,
                         std::vector<float>* out, std::vector<size_t>* shape = nullptr) {
    std::vector<uint8_t> buf;
    if (!read_file(path, &buf)) return false;

    std::vector<uint8_t> entry;
    if (!ZipExtractStored(buf, key + ".npy", &entry)) return false;
    return npy_parse_f32(entry.data(), entry.size(), out, shape);
}

// Writes a float32 .npy file in C order.
// The header, counting the 6 magic bytes, the 2 version bytes and the 2 length bytes, must be
// padded so the total is a multiple of 64 and must end with a newline; numpy rejects a file
// that breaks either rule.
//
// in:  path   destination file path
//      data   values to write
//      shape  dimensions to record
// out: return true on success
inline bool write_npy_f32(const std::string& path, const std::vector<float>& data,
                          const std::vector<size_t>& shape) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    std::string shape_str;
    for (size_t i = 0; i < shape.size(); i++) {
        if (i) shape_str += ", ";
        shape_str += std::to_string(shape[i]);
    }
    if (shape.size() == 1) shape_str += ",";   // a one-element tuple needs the trailing comma

    char header[256];
    const int n = snprintf(header, sizeof(header),
                           "{'descr': '<f4', 'fortran_order': False, 'shape': (%s), }",
                           shape_str.c_str());
    const int total = 10 + n + 1;
    const int pad = (64 - (total % 64)) % 64;
    std::string h(header, static_cast<size_t>(n));
    h.append(static_cast<size_t>(pad), ' ');
    h += '\n';
    const uint16_t hlen = static_cast<uint16_t>(h.size());

    bool ok = fwrite("\x93NUMPY", 1, 6, f) == 6;
    ok = ok && fputc(1, f) != EOF && fputc(0, f) != EOF;
    ok = ok && fwrite(&hlen, 2, 1, f) == 1;
    ok = ok && fwrite(h.data(), 1, h.size(), f) == h.size();
    if (!data.empty()) ok = ok && fwrite(data.data(), 4, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

}  // namespace pi05

#endif  // PI05_NPY_IO_H
