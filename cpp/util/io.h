#pragma once

// include json-related io
#include "json.h"

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

inline std::string readFile(const std::string& path) {
    std::ifstream in(path);
    std::stringstream strStream;
    strStream << in.rdbuf(); //read the file
    std::string params = strStream.str();
    return params;
}

inline bool checkStrPresent(std::istream& in, const std::string& expected) {
    std::string actual(expected.size(), ' ');
    if (!in.read(actual.data(), expected.size())) {
        return false;
    }
    return actual == expected;
}

inline bool couldRead(std::istream& in, void* obj, std::size_t size, const std::string& enclosing) {
    if (!checkStrPresent(in, enclosing + "{")) {
        return false;
    }
    if (!in.read((char*)obj, size)) {
        return false;
    }
    return checkStrPresent(in, "}");
}

inline void writeEnclosed(std::ostream& out, void* obj, std::size_t size, std::string enclosing) {
    enclosing += "{";
    out.write(enclosing.data(), enclosing.size());
    out.write((char*)obj, size);
    out.write("}", 1);
}
