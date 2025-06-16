// @Note: This is a minimal Lua-like interpreter in C++ (~By Daniel)
// It supports basic expressions, variables, functions, and control flow
// This file contains both the AST node definitions and the interpreter logic

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <variant>
#include <memory>
#include <stdexcept>

// @Note: Forward declarations
struct Environment;
struct Expression;
struct Statement;
struct FunctionNode;
using EnvironmentPtr = std::shared_ptr<Environment>;
using ExpressionPtr = std::shared_ptr<Expression>;
using StatementPtr = std::shared_ptr<Statement>;

// @Note: Value type used to hold runtime values
struct Value {
    enum class Type { Nil, Boolean, Number, String, Function, Table } type;

    std::variant<std::nullptr_t, bool, double, std::string, FunctionNode*> data;

    Value() : type(Type::Nil), data(nullptr) {}
    Value(std::nullptr_t) : type(Type::Nil), data(nullptr) {}
    Value(bool b) : type(Type::Boolean), data(b) {}
    Value(double n) : type(Type::Number), data(n) {}
    Value(const std::string& s) : type(Type::String), data(s) {}
    Value(FunctionNode* f) : type(Type::Function), data(f) {}

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

// @Note: Environment for variable/function lookup
struct Environment {
    std::unordered_map<std::string, Value> values;
    EnvironmentPtr parent;

    Environment(EnvironmentPtr parent = nullptr) : parent(parent) {}

    void Define(const std::string& name, const Value& val) {
        values[name] = val;
    }

    Value Get(const std::string& name) {
        if (values.count(name)) return values[name];
        if (parent) return parent->Get(name);
        throw std::runtime_error("Undefined variable: " + name);
    }

    void Assign(const std::string& name, const Value& val) {
        if (values.count(name)) values[name] = val;
        else if (parent) parent->Assign(name, val);
        else throw std::runtime_error("Undefined variable: " + name);
    }
};

// @Note: Base classes
struct Expression {
    virtual Value Evaluate(EnvironmentPtr env) = 0;
    virtual ~Expression() = default;
};

struct Statement {
    virtual void Execute(EnvironmentPtr env) = 0;
    virtual ~Statement() = default;
};

// @Note: Expression types
struct NumberExpr : Expression {
    double value;
    NumberExpr(double v) : value(v) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(value); }
};

struct StringExpr : Expression {
    std::string value;
    StringExpr(const std::string& v) : value(v) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(value); }
};

struct BooleanExpr : Expression {
    bool value;
    BooleanExpr(bool v) : value(v) {}
    Value Evaluate(EnvironmentPtr env) override { return Value(value); }
};

struct VariableExpr : Expression {
    std::string name;
    VariableExpr(const std::string& n) : name(n) {}
    Value Evaluate(EnvironmentPtr env) override { return env->Get(name); }
};

struct BinaryExpr : Expression {
    enum class Operator { Add, Sub, Mul, Div, Equal } op;
    ExpressionPtr left, right;

    BinaryExpr(Operator op, ExpressionPtr l, ExpressionPtr r) : op(op), left(l), right(r) {}

    Value Evaluate(EnvironmentPtr env) override {
        Value lv = left->Evaluate(env);
        Value rv = right->Evaluate(env);

        if (op == Operator::Add) return Value(std::get<double>(lv.data) + std::get<double>(rv.data));
        if (op == Operator::Sub) return Value(std::get<double>(lv.data) - std::get<double>(rv.data));
        if (op == Operator::Mul) return Value(std::get<double>(lv.data) * std::get<double>(rv.data));
        if (op == Operator::Div) return Value(std::get<double>(lv.data) / std::get<double>(rv.data));
        if (op == Operator::Equal) return Value(lv.ToString() == rv.ToString());

        return Value();
    }
};

// @Note: Function node
struct FunctionNode {
    std::vector<std::string> params;
    std::vector<StatementPtr> body;
    EnvironmentPtr closure;

    FunctionNode(const std::vector<std::string>& p, const std::vector<StatementPtr>& b, EnvironmentPtr env)
        : params(p), body(b), closure(env) {}

    Value Call(std::vector<Value> args) {
        auto local = std::make_shared<Environment>(closure);
        for (size_t i = 0; i < params.size(); ++i) {
            local->Define(params[i], i < args.size() ? args[i] : Value());
        }
        for (auto& stmt : body) stmt->Execute(local);
        return Value();
    }
};

// @Note: Function call expression
struct CallExpr : Expression {
    ExpressionPtr callee;
    std::vector<ExpressionPtr> arguments;

    CallExpr(ExpressionPtr c, std::vector<ExpressionPtr> a) : callee(c), arguments(a) {}

    Value Evaluate(EnvironmentPtr env) override {
        Value fnVal = callee->Evaluate(env);
        if (fnVal.type != Value::Type::Function)
            throw std::runtime_error("Trying to call non-function");

        std::vector<Value> args;
        for (auto& arg : arguments) args.push_back(arg->Evaluate(env));

        return std::get<FunctionNode*>(fnVal.data)->Call(args);
    }
};

// @Note: Statement types
struct ExpressionStmt : Statement {
    ExpressionPtr expr;
    ExpressionStmt(ExpressionPtr e) : expr(e) {}
    void Execute(EnvironmentPtr env) override { expr->Evaluate(env); }
};

struct PrintStmt : Statement {
    ExpressionPtr expr;
    PrintStmt(ExpressionPtr e) : expr(e) {}
    void Execute(EnvironmentPtr env) override {
        std::cout << expr->Evaluate(env).ToString() << std::endl;
    }
};

struct VarStmt : Statement {
    std::string name;
    ExpressionPtr initializer;
    VarStmt(const std::string& n, ExpressionPtr init) : name(n), initializer(init) {}
    void Execute(EnvironmentPtr env) override {
        env->Define(name, initializer->Evaluate(env));
    }
};

struct BlockStmt : Statement {
    std::vector<StatementPtr> statements;
    BlockStmt(const std::vector<StatementPtr>& stmts) : statements(stmts) {}
    void Execute(EnvironmentPtr env) override {
        auto local = std::make_shared<Environment>(env);
        for (auto& stmt : statements) stmt->Execute(local);
    }
};

struct FunctionStmt : Statement {
    std::string name;
    std::vector<std::string> params;
    std::vector<StatementPtr> body;

    FunctionStmt(const std::string& n, const std::vector<std::string>& p, const std::vector<StatementPtr>& b)
        : name(n), params(p), body(b) {}

    void Execute(EnvironmentPtr env) override {
        auto fn = new FunctionNode(params, body, env);
        env->Define(name, Value(fn));
    }
};

// @Note: Entry point
int main() {
    auto env = std::make_shared<Environment>();
    std::vector<StatementPtr> program = {
        std::make_shared<VarStmt>("x", std::make_shared<NumberExpr>(10)),
        std::make_shared<PrintStmt>(std::make_shared<VariableExpr>("x")),
    };

    for (auto& stmt : program) stmt->Execute(env);
    return 0;
}
