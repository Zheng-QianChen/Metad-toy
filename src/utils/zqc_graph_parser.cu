#include "zqc_graph.cuh"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <fmt/printf.h>

namespace {
GraphExpr call_expression(const std::string& name, std::vector<GraphExpr> args,
                          std::size_t position) {
    return {GraphExpr::Kind::Call, name, 0.0, std::move(args), position};
}
}

GraphExprParser::GraphExprParser(std::string text) : text_(std::move(text)) {}

void GraphExprParser::fail(const std::string& message) const {
    throw std::runtime_error(fmt::sprintf("Parse error at position %zu: %s", pos_, message));
}

void GraphExprParser::skip_spaces() {
    while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_])))
        ++pos_;
}

bool GraphExprParser::take(char expected) {
    skip_spaces();
    if (pos_ < text_.size() && text_[pos_] == expected) {
        ++pos_;
        return true;
    }
    return false;
}

void GraphExprParser::expect(char expected) {
    if (!take(expected)) fail(fmt::sprintf("expected '%c'", expected));
}

std::string GraphExprParser::read_identifier() {
    skip_spaces();
    if (pos_ == text_.size()) fail("expected identifier");
    unsigned char ch = static_cast<unsigned char>(text_[pos_]);
    if (!std::isalpha(ch) && ch != '_') fail("expected identifier");
    const std::size_t begin = pos_++;
    while (pos_ < text_.size()) {
        ch = static_cast<unsigned char>(text_[pos_]);
        if (!std::isalnum(ch) && ch != '_') break;
        ++pos_;
    }
    return text_.substr(begin, pos_ - begin);
}

GraphExpr GraphExprParser::read_number() {
    const std::size_t begin = pos_;
    const char* start = text_.c_str() + pos_;
    char* end = nullptr;
    const double value = std::strtod(start, &end);
    if (end == start || !std::isfinite(value)) fail("expected a finite number");
    pos_ += static_cast<std::size_t>(end - start);
    return {GraphExpr::Kind::Number, {}, value, {}, begin};
}

GraphExpr GraphExprParser::parse() {
    GraphExpr root = parse_sum();
    // The outer suffix is syntax only; its meaning comes from the node registry.
    if (take('.')) {
        const std::size_t begin = pos_ - 1;
        std::string name = "." + read_identifier();
        root = call_expression(name, {std::move(root)}, begin);
    }
    skip_spaces();
    if (pos_ != text_.size()) fail("unexpected trailing input");
    return root;
}

GraphExpr GraphExprParser::parse_sum() {
    GraphExpr lhs = parse_product();
    for (;;) {
        skip_spaces();
        const std::size_t begin = pos_;
        if (take('+')) {
            GraphExpr rhs = parse_product();
            lhs = call_expression("+", {std::move(lhs), std::move(rhs)}, begin);
        } else if (take('-')) {
            GraphExpr rhs = parse_product();
            lhs = call_expression("-", {std::move(lhs), std::move(rhs)}, begin);
        } else {
            return lhs;
        }
    }
}

GraphExpr GraphExprParser::parse_product() {
    GraphExpr lhs = parse_unary();
    while (take('*')) {
        const std::size_t begin = pos_ - 1;
        GraphExpr rhs = parse_unary();
        lhs = call_expression("*", {std::move(lhs), std::move(rhs)}, begin);
    }
    return lhs;
}

GraphExpr GraphExprParser::parse_unary() {
    if (take('+')) return parse_unary();
    if (take('-')) {
        const std::size_t begin = pos_ - 1;
        GraphExpr input = parse_unary();
        if (input.kind == GraphExpr::Kind::Number) {
            input.number = -input.number;
            input.position = begin;
            return input;
        }
        return call_expression("neg", {std::move(input)}, begin);
    }
    return parse_primary();
}

GraphExpr GraphExprParser::parse_primary() {
    if (take('(')) {
        GraphExpr expression = parse_sum();
        expect(')');
        return expression;
    }
    skip_spaces();
    if (pos_ == text_.size()) fail("expected expression");
    if (std::isdigit(static_cast<unsigned char>(text_[pos_])) || text_[pos_] == '.')
        return read_number();

    const std::size_t begin = pos_;
    const std::string name = read_identifier();
    if (!take('(')) return {GraphExpr::Kind::Symbol, name, 0.0, {}, begin};
    std::vector<GraphExpr> args;
    if (!take(')')) {
        do {
            args.push_back(parse_sum());
        } while (take(','));
        expect(')');
    }
    return call_expression(name, std::move(args), begin);
}
