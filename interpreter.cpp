#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cctype>
#include <sstream>
#include <variant>
#include <unordered_set>
#include <optional>

enum class TokenType {
    Identifier, Keyword, Number, String,
    Plus, Minus, Star, Slash, Percent,
    Equal, DoubleEqual, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    LeftParen, RightParen, LeftBrace, RightBrace, Comma, Dot, Semicolon,
    Assign,
    And, Or, Not,
    If, Then, Else, ElseIf, End,
    Function, Local, While, For, In, Do, Return, Break,
    Nil, True, False,
    Eof, Invalid
};

struct Token {
    TokenType Type;
    std::string Lexeme;
    int Line;

    Token(TokenType type, const std::string& lexeme, int line)
        : Type(type), Lexeme(lexeme), Line(line) {
    }
};

class Lexer {
private:
    std::string Source;
    size_t Pos = 0;
    int Line = 1;

    std::unordered_map<std::string, TokenType> Keywords = {
        {"and", TokenType::And}, {"or", TokenType::Or}, {"not", TokenType::Not},
        {"if", TokenType::If}, {"then", TokenType::Then},
        {"else", TokenType::Else}, {"elseif", TokenType::ElseIf},
        {"end", TokenType::End}, {"function", TokenType::Function},
        {"local", TokenType::Local}, {"while", TokenType::While},
        {"for", TokenType::For}, {"in", TokenType::In},
        {"do", TokenType::Do}, {"return", TokenType::Return},
        {"break", TokenType::Break}, {"nil", TokenType::Nil},
        {"true", TokenType::True}, {"false", TokenType::False}
    };

    char Peek() const {
        return Pos >= Source.length() ? '\0' : Source[Pos];
    }

    char PeekNext() const {
        return (Pos + 1) >= Source.length() ? '\0' : Source[Pos + 1];
    }

    char Advance() {
        return Source[Pos++];
    }

    bool Match(char expected) {
        if (Peek() != expected) return false;
        Pos++;
        return true;
    }

    void SkipWhitespace() {
        while (true) {
            char c = Peek();
            if (c == ' ' || c == '\r' || c == '\t') {
                Advance();
            }
            else if (c == '\n') {
                Line++;
                Advance();
            }
            else if (c == '-' && PeekNext() == '-') {
                Advance(); Advance();
                while (Peek() != '\n' && Peek() != '\0') Advance();
            }
            else break;
        }
    }

    Token StringLiteral() {
        std::string value;
        Advance();
        while (Peek() != '"' && Peek() != '\0') {
            if (Peek() == '\\') {
                Advance();
                char esc = Peek();
                switch (esc) {
                case 'n': value += '\n'; break;
                case 't': value += '\t'; break;
                case '"': value += '"'; break;
                default: value += esc; break;
                }
                Advance();
            }
            else {
                value += Advance();
            }
        }
        if (Peek() == '"') Advance();
        return Token(TokenType::String, value, Line);
    }

    Token NumberLiteral() {
        std::string value;
        while (isdigit(Peek()) || Peek() == '.') {
            value += Advance();
        }
        return Token(TokenType::Number, value, Line);
    }

    Token IdentifierOrKeyword() {
        std::string value;
        while (isalnum(Peek()) || Peek() == '_') {
            value += Advance();
        }
        auto it = Keywords.find(value);
        if (it != Keywords.end()) {
            return Token(it->second, value, Line);
        }
        return Token(TokenType::Identifier, value, Line);
    }

public:
    explicit Lexer(const std::string& source) : Source(source) {}

    std::vector<Token> Tokenize() {
        std::vector<Token> tokens;
        while (Pos < Source.size()) {
            SkipWhitespace();
            char c = Peek();
            if (c == '\0') break;

            if (isdigit(c)) {
                tokens.push_back(NumberLiteral());
            }
            else if (isalpha(c) || c == '_') {
                tokens.push_back(IdentifierOrKeyword());
            }
            else {
                switch (c) {
                case '+': tokens.emplace_back(TokenType::Plus, "+", Line); Advance(); break;
                case '-': tokens.emplace_back(TokenType::Minus, "-", Line); Advance(); break;
                case '*': tokens.emplace_back(TokenType::Star, "*", Line); Advance(); break;
                case '/': tokens.emplace_back(TokenType::Slash, "/", Line); Advance(); break;
                case '%': tokens.emplace_back(TokenType::Percent, "%", Line); Advance(); break;
                case '(': tokens.emplace_back(TokenType::LeftParen, "(", Line); Advance(); break;
                case ')': tokens.emplace_back(TokenType::RightParen, ")", Line); Advance(); break;
                case '{': tokens.emplace_back(TokenType::LeftBrace, "{", Line); Advance(); break;
                case '}': tokens.emplace_back(TokenType::RightBrace, "}", Line); Advance(); break;
                case ',': tokens.emplace_back(TokenType::Comma, ",", Line); Advance(); break;
                case '.': tokens.emplace_back(TokenType::Dot, ".", Line); Advance(); break;
                case ';': tokens.emplace_back(TokenType::Semicolon, ";", Line); Advance(); break;
                case '=':
                    Advance();
                    if (Match('=')) tokens.emplace_back(TokenType::DoubleEqual, "==", Line);
                    else tokens.emplace_back(TokenType::Assign, "=", Line);
                    break;
                case '<':
                    Advance();
                    if (Match('=')) tokens.emplace_back(TokenType::LessEqual, "<=", Line);
                    else tokens.emplace_back(TokenType::Less, "<", Line);
                    break;
                case '>':
                    Advance();
                    if (Match('=')) tokens.emplace_back(TokenType::GreaterEqual, ">=", Line);
                    else tokens.emplace_back(TokenType::Greater, ">", Line);
                    break;
                case '!':
                    Advance();
                    if (Match('=')) tokens.emplace_back(TokenType::NotEqual, "!=", Line);
                    else tokens.emplace_back(TokenType::Invalid, "!", Line);
                    break;
                case '"':
                    tokens.push_back(StringLiteral());
                    break;
                default:
                    tokens.emplace_back(TokenType::Invalid, std::string(1, c), Line);
                    Advance();
                    break;
                }
            }
        }
        tokens.emplace_back(TokenType::Eof, "", Line);
        return tokens;
    }
};

struct AstNode {
    virtual ~AstNode() = default;
};

using AstNodePtr = std::unique_ptr<AstNode>;

struct Value;

using EnvironmentPtr = std::shared_ptr<struct Environment>;

struct Value {

    enum class Type { Nil, Boolean, Number, String, Function, Table } type;

    std::variant<std::nullptr_t, bool, double, std::string, struct FunctionNode*, struct Table*> data;

    Value() : type(Type::Nil), data(nullptr) {}
    Value(std::nullptr_t) : type(Type::Nil), data(nullptr) {}
    Value(bool b) : type(Type::Boolean), data(b) {}
    Value(double n) : type(Type::Number), data(n) {}
    Value(const std::string& s) : type(Type::String), data(s) {}
    Value(FunctionNode* f) : type(Type::Function), data(f) {}
    Value(struct Table* t) : type(Type::Table), data(t) {}

    bool IsTrue() const {
        if (type == Type::Nil) return false;
        if (type == Type::Boolean) return std::get<bool>(data);
        return true;
    }

    std::string ToString() const {
        switch (type) {
        case Type::Nil: return "nil";
        case Type::Boolean: return std::get<bool>(data) ? "true" : "false";
        case Type::Number: return std::to_string(std::get<double>(data));
        case Type::String: return std::get<std::string>(data);
        case Type::Function: return "<function>";
        case Type::Table: return "<table>";
        }
        return "<unknown>";
    }
};

struct Table {
    std::unordered_map<std::string, Value> Fields;
};

struct Environment {
    EnvironmentPtr Parent;
    std::unordered_map<std::string, Value> Variables;

    explicit Environment(EnvironmentPtr parent = nullptr) : Parent(parent) {}

    void Set(const std::string& name, Value val) {

        if (Variables.find(name) != Variables.end()) {
            Variables[name] = val;
        }
        else if (Parent) {
            Parent->Set(name, val);
        }
        else {
            Variables[name] = val;
        }
    }

    void Declare(const std::string& name, Value val) {
        Variables[name] = val;
    }

    std::optional<Value> Get(const std::string& name) {
        auto it = Variables.find(name);
        if (it != Variables.end()) return it->second;
        if (Parent) return Parent->Get(name);
        return {};
    }
};

struct Expression : AstNode {
    virtual Value Evaluate(EnvironmentPtr env) = 0;
};

using ExpressionPtr = std::unique_ptr<Expression>;

struct NumberExpr : Expression {
    double number;

    NumberExpr(double value) : number(value) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(number); }
};

struct StringExpr : Expression {
    std::string str;

    StringExpr(const std::string& value) : str(value) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(str); }
};

struct BooleanExpr : Expression {
    bool boolean;

    BooleanExpr(bool value) : boolean(value) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(boolean); }
};

struct NilExpr : Expression {
    Value Evaluate(EnvironmentPtr env) override { return Value(nullptr); }
};

struct VariableExpr : Expression {
    std::string Name;

    VariableExpr(const std::string& name) : Name(name) {}
    Value Evaluate(EnvironmentPtr env) override {
        auto val = env->Get(Name);
        if (val) return *val;
        std::cerr << "Runtime error: variable '" << Name << "' is nil or undefined.\n";
        return Value(nullptr);
    }
};

struct BinaryExpr : Expression {
    enum class OpType { Add, Sub, Mul, Div, Mod, Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual, And, Or };
    OpType Op;
    ExpressionPtr Left;
    ExpressionPtr Right;

    BinaryExpr(OpType op, ExpressionPtr left, ExpressionPtr right)
        : Op(op), Left(std::move(left)), Right(std::move(right)) {
    }

    Value Evaluate(EnvironmentPtr env) override {
        Value leftVal = Left->Evaluate(env);
        Value rightVal = Right->Evaluate(env);

        if (Op == OpType::Add) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number) {
                return Value(std::get<double>(leftVal.data) + std::get<double>(rightVal.data));
            }
        }
        else if (Op == OpType::Sub) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number) {
                return Value(std::get<double>(leftVal.data) - std::get<double>(rightVal.data));
            }
        }
        else if (Op == OpType::Mul) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number) {
                return Value(std::get<double>(leftVal.data) * std::get<double>(rightVal.data));
            }
        }
        else if (Op == OpType::Div) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number) {
                if (std::get<double>(rightVal.data) == 0) {
                    std::cerr << "Runtime error: division by zero\n";
                    return Value(nullptr);
                }
                return Value(std::get<double>(leftVal.data) / std::get<double>(rightVal.data));
            }
        }
        else if (Op == OpType::Equal) {
            return Value(leftVal.ToString() == rightVal.ToString());
        }
        else if (Op == OpType::NotEqual) {
            return Value(leftVal.ToString() != rightVal.ToString());
        }
        else if (Op == OpType::Less) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number)
                return Value(std::get<double>(leftVal.data) < std::get<double>(rightVal.data));
        }
        else if (Op == OpType::LessEqual) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number)
                return Value(std::get<double>(leftVal.data) <= std::get<double>(rightVal.data));
        }
        else if (Op == OpType::Greater) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number)
                return Value(std::get<double>(leftVal.data) > std::get<double>(rightVal.data));
        }
        else if (Op == OpType::GreaterEqual) {
            if (leftVal.type == Value::Type::Number && rightVal.type == Value::Type::Number)
                return Value(std::get<double>(leftVal.data) >= std::get<double>(rightVal.data));
        }
        else if (Op == OpType::And) {
            return Value(leftVal.IsTrue() && rightVal.IsTrue());
        }
        else if (Op == OpType::Or) {
            return Value(leftVal.IsTrue() || rightVal.IsTrue());
        }
        std::cerr << "Runtime error: unsupported binary operation\n";
        return Value(nullptr);
    }
};

struct Statement : AstNode {
    virtual void Execute(EnvironmentPtr env) = 0;
};

using StatementPtr = std::unique_ptr<Statement>;

struct BlockStmt : Statement {
    std::vector<StatementPtr> Statements;
    void Execute(EnvironmentPtr env) override {
        for (auto& stmt : Statements) {
            stmt->Execute(env);
        }
    }
};

struct ExprStmt : Statement {
    ExpressionPtr Expr;
    ExprStmt(ExpressionPtr expr) : Expr(std::move(expr)) {}
    void Execute(EnvironmentPtr env) override {
        Expr->Evaluate(env);
    }
};

struct AssignStmt : Statement {
    std::string VariableName;
    ExpressionPtr ValueExpr;

    AssignStmt(std::string varName, ExpressionPtr val) : VariableName(std::move(varName)), ValueExpr(std::move(val)) {}

    void Execute(EnvironmentPtr env) override {
        Value val = ValueExpr->Evaluate(env);
        env->Set(VariableName, val);
    }
};

struct LocalAssignStmt : Statement {
    std::string VariableName;
    ExpressionPtr ValueExpr;

    LocalAssignStmt(std::string varName, ExpressionPtr val) : VariableName(std::move(varName)), ValueExpr(std::move(val)) {}

    void Execute(EnvironmentPtr env) override {
        Value val = ValueExpr->Evaluate(env);
        env->Declare(VariableName, val);
    }
};

struct IfStmt : Statement {
    ExpressionPtr Condition;
    StatementPtr ThenBlock;
    StatementPtr ElseBlock;

    IfStmt(ExpressionPtr cond, StatementPtr thenBlk, StatementPtr elseBlk = nullptr)
        : Condition(std::move(cond)), ThenBlock(std::move(thenBlk)), ElseBlock(std::move(elseBlk)) {
    }

    void Execute(EnvironmentPtr env) override {
        if (Condition->Evaluate(env).IsTrue()) {
            ThenBlock->Execute(env);
        }
        else if (ElseBlock) {
            ElseBlock->Execute(env);
        }
    }
};

class Parser {
private:
    const std::vector<Token>& Tokens;
    size_t Current = 0;

    const Token& Peek() const {
        if (Current >= Tokens.size()) return Tokens.back();
        return Tokens[Current];
    }

    const Token& Advance() {
        if (Current < Tokens.size()) Current++;
        return Previous();
    }

    const Token& Previous() const {
        return Tokens[Current - 1];
    }

    bool Match(std::initializer_list<TokenType> types) {
        for (auto type : types) {
            if (Check(type)) {
                Advance();
                return true;
            }
        }
        return false;
    }

    bool Check(TokenType type) const {
        if (Current >= Tokens.size()) return false;
        return Tokens[Current].Type == type;
    }

    void Consume(TokenType type, const std::string& message) {
        if (Check(type)) {
            Advance();
            return;
        }
        throw std::runtime_error("Parser error: Expected " + message + " at line " + std::to_string(Peek().Line));
    }

    ExpressionPtr ParseExpression() {
        return ParseOr();
    }

    ExpressionPtr ParsePrimary() {
        const Token& token = Peek();
        if (Match({ TokenType::Number })) {
            return std::make_unique<NumberExpr>(std::stod(Previous().Lexeme));
        }
        if (Match({ TokenType::String })) {
            return std::make_unique<StringExpr>(Previous().Lexeme);
        }
        if (Match({ TokenType::True })) {
            return std::make_unique<BooleanExpr>(true);
        }
        if (Match({ TokenType::False })) {
            return std::make_unique<BooleanExpr>(false);
        }
        if (Match({ TokenType::Nil })) {
            return std::make_unique<NilExpr>();
        }
        if (Match({ TokenType::Identifier })) {
            return std::make_unique<VariableExpr>(Previous().Lexeme);
        }
        if (Match({ TokenType::LeftParen })) {
            ExpressionPtr expr = ParseExpression();
            Consume(TokenType::RightParen, "')'");
            return expr;
        }

        throw std::runtime_error("Parser error: Unexpected token '" + token.Lexeme + "' at line " + std::to_string(token.Line));
    }

    ExpressionPtr ParseUnary() {
        if (Match({ TokenType::Minus, TokenType::Not })) {
            Token op = Previous();
            ExpressionPtr right = ParseUnary();

            if (op.Type == TokenType::Minus) {
                return std::make_unique<BinaryExpr>(BinaryExpr::OpType::Sub,
                    std::make_unique<NumberExpr>(0), std::move(right));
            }
            else if (op.Type == TokenType::Not) {

            }
        }
        return ParsePrimary();
    }

    ExpressionPtr ParseMultiplicative() {
        ExpressionPtr expr = ParseUnary();
        while (Match({ TokenType::Star, TokenType::Slash, TokenType::Percent })) {
            Token op = Previous();
            ExpressionPtr right = ParseUnary();
            BinaryExpr::OpType opType;
            if (op.Type == TokenType::Star) opType = BinaryExpr::OpType::Mul;
            else if (op.Type == TokenType::Slash) opType = BinaryExpr::OpType::Div;
            else opType = BinaryExpr::OpType::Mod;
            expr = std::make_unique<BinaryExpr>(opType, std::move(expr), std::move(right));
        }
        return expr;
    }

    ExpressionPtr ParseAdditive() {
        ExpressionPtr expr = ParseMultiplicative();
        while (Match({ TokenType::Plus, TokenType::Minus })) {
            Token op = Previous();
            ExpressionPtr right = ParseMultiplicative();
            BinaryExpr::OpType opType = (op.Type == TokenType::Plus) ? BinaryExpr::OpType::Add : BinaryExpr::OpType::Sub;
            expr = std::make_unique<BinaryExpr>(opType, std::move(expr), std::move(right));
        }
        return expr;
    }

    ExpressionPtr ParseComparison() {
        ExpressionPtr expr = ParseAdditive();
        while (Match({ TokenType::Less, TokenType::LessEqual, TokenType::Greater, TokenType::GreaterEqual, TokenType::DoubleEqual, TokenType::NotEqual })) {
            Token op = Previous();
            ExpressionPtr right = ParseAdditive();
            BinaryExpr::OpType opType;
            switch (op.Type) {
            case TokenType::Less: opType = BinaryExpr::OpType::Less; break;
            case TokenType::LessEqual: opType = BinaryExpr::OpType::LessEqual; break;
            case TokenType::Greater: opType = BinaryExpr::OpType::Greater; break;
            case TokenType::GreaterEqual: opType = BinaryExpr::OpType::GreaterEqual; break;
            case TokenType::DoubleEqual: opType = BinaryExpr::OpType::Equal; break;
            case TokenType::NotEqual: opType = BinaryExpr::OpType::NotEqual; break;
            default: throw std::runtime_error("Parser error: Unknown comparison operator");
            }
            expr = std::make_unique<BinaryExpr>(opType, std::move(expr), std::move(right));
        }
        return expr;
    }

    ExpressionPtr ParseAnd() {
        ExpressionPtr expr = ParseComparison();
        while (Match({ TokenType::And })) {
            ExpressionPtr right = ParseComparison();
            expr = std::make_unique<BinaryExpr>(BinaryExpr::OpType::And, std::move(expr), std::move(right));
        }
        return expr;
    }

    ExpressionPtr ParseOr() {
        ExpressionPtr expr = ParseAnd();
        while (Match({ TokenType::Or })) {
            ExpressionPtr right = ParseAnd();
            expr = std::make_unique<BinaryExpr>(BinaryExpr::OpType::Or, std::move(expr), std::move(right));
        }
        return expr;
    }

    StatementPtr ParseStatement() {
        if (Match({ TokenType::If })) {
            ExpressionPtr condition = ParseExpression();
            StatementPtr thenBlock = ParseStatement();
            StatementPtr elseBlock = nullptr;
            if (Match({ TokenType::Else })) {
                elseBlock = ParseStatement();
            }
            return std::make_unique<IfStmt>(std::move(condition), std::move(thenBlock), std::move(elseBlock));
        }
        else if (Match({ TokenType::LeftBrace })) {
            auto block = std::make_unique<BlockStmt>();
            while (!Check(TokenType::RightBrace) && Current < Tokens.size()) {
                block->Statements.push_back(ParseStatement());
            }
            Consume(TokenType::RightBrace, "'}'");
            return block;
        }
        else if (Match({ TokenType::Local })) {
            if (!Check(TokenType::Identifier)) throw std::runtime_error("Parser error: Expected variable name after 'local'");
            std::string varName = Peek().Lexeme;
            Advance();
            Consume(TokenType::Equal, "'=' after variable name");
            ExpressionPtr valExpr = ParseExpression();
            return std::make_unique<LocalAssignStmt>(varName, std::move(valExpr));
        }
        else if (Match({ TokenType::Identifier })) {
            std::string varName = Previous().Lexeme;
            if (Match({ TokenType::Equal })) {
                ExpressionPtr valExpr = ParseExpression();
                return std::make_unique<AssignStmt>(varName, std::move(valExpr));
            }
            throw std::runtime_error("Parser error: Unexpected token after identifier");
        }
        else if (Match({ TokenType::Semicolon })) {
            return std::make_unique<BlockStmt>();
        }
        else {
            ExpressionPtr expr = ParseExpression();
            Consume(TokenType::Semicolon, "';' after expression");
            return std::make_unique<ExprStmt>(std::move(expr));
        }
    }

public:
    Parser(const std::vector<Token>& tokens) : Tokens(tokens) {}

    std::vector<StatementPtr> Parse() {
        std::vector<StatementPtr> statements;
        while (Current < Tokens.size()) {
            statements.push_back(ParseStatement());
        }
        return statements;
    }
};

class Interpreter {
private:
    EnvironmentPtr GlobalEnv = std::make_shared<Environment>();

public:
    void Interpret(const std::vector<StatementPtr>& statements) {
        try {
            for (const auto& stmt : statements) {
                stmt->Execute(GlobalEnv);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Runtime error: " << e.what() << "\n";
        }
    }
};

int main() {
    std::string source = R"(
        local x = 10;
        local y = 20;
        local z = x + y * 2;
        if (z > 40) {
            x = x + 1;
        } else {
            x = x - 1;
        }
    )";

    Lexer lexer(source);
    std::vector<Token> tokens = lexer.Tokenize();

    Parser parser(tokens);
    auto statements = parser.Parse();

    Interpreter interpreter;
    interpreter.Interpret(statements);

    return 0;
}
