# Agent Fuzz - AFL++ 自动化模糊测试工作流

基于 LangChain 框架的 AFL++ 模糊测试自动化工作流系统。

## 项目简介

Agent Fuzz 是一个使用 LangChain 框架实现的 AFL++ 模糊测试自动化工作流。它将 AFL++ 模糊测试的完整流程（从目标文件分析到结果分析）表示为可执行的自动化流程，支持使用免费的 API 进行测试。

## 功能特性

- ✅ **完整工作流**: 8 个主要阶段的自动化流程
- ✅ **LangChain 集成**: 使用 LangChain 框架实现智能工作流
- ✅ **免费 API 支持**: 支持 Hugging Face Hub 等免费 API
- ✅ **灵活扩展**: 易于添加新工具和工作流阶段
- ✅ **状态管理**: 完整的工作流状态跟踪
- ✅ **错误处理**: 健壮的错误处理和恢复机制

## 工作流阶段

1. **目标分析** (Analyze) - 分析 .o 文件，提取函数和符号
2. **生成 Harness** (Generate Harness) - 自动生成 fuzz harness 代码
3. **环境准备** (Prepare Environment) - 准备 Docker 容器环境
4. **编译** (Compile) - 使用 AFL++ 编译器编译
5. **创建种子** (Create Seeds) - 生成测试种子文件
6. **功能测试** (Functional Test) - 验证程序功能
7. **覆盖率测试** (Coverage Test) - 检查代码覆盖率
8. **模糊测试** (Fuzzing) - 启动 AFL++ 模糊测试
9. **结果分析** (Analysis) - 分析测试结果

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API (可选)

#### 使用 Hugging Face 免费 API

```bash
# Windows PowerShell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"

# Linux/Mac
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

获取 token: https://huggingface.co/settings/tokens

#### 使用本地 LLM (Ollama)

```bash
# 安装并运行 Ollama
ollama run llama2

# 设置环境变量
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```

### 3. 运行测试

```bash
python test_workflow.py
```

### 4. 使用工作流

```python
from afl_workflow_langchain import AFLWorkflowLangChain

# 创建工作流实例
workflow = AFLWorkflowLangChain(use_free_api=True)

# 执行完整工作流
results = workflow.execute_full_workflow("vul_bn_exp.o")
```

## 项目结构

```
agent_fuzz/
├── afl_workflow_langchain.py  # 主工作流实现
├── test_workflow.py           # 测试脚本
├── requirements.txt           # 依赖列表
├── README.md                  # 项目说明
├── README_LANGCHAIN.md       # LangChain 详细文档
└── .gitignore                # Git 忽略文件
```

## 技术栈

- **LangChain**: 工作流框架
- **Python 3.8+**: 编程语言
- **Hugging Face Hub**: 免费 LLM API
- **AFL++**: 模糊测试工具

## 文档

- [LangChain 工作流详细文档](README_LANGCHAIN.md)
- [AFL++ 完整指南](../AFL++_Fuzzing_Complete_Guide.md)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request!

## 相关资源

- [LangChain 文档](https://python.langchain.com/)
- [AFL++ 文档](https://github.com/AFLplusplus/AFLplusplus)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

