# AFL++ 工作流 LangChain 实现

这个项目将 AFL++ 模糊测试的完整工作流用 LangChain 框架表示，支持使用免费的 API 进行测试。

## 功能特性

- ✅ 使用 LangChain 框架实现完整的工作流
- ✅ 支持 8 个主要工作流阶段
- ✅ 支持免费 API (Hugging Face Hub)
- ✅ 支持 OpenAI 兼容 API (LocalAI, Ollama 等)
- ✅ 完整的状态管理和错误处理
- ✅ 可扩展的工具系统

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

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API (可选)

#### 选项 A: 使用 Hugging Face 免费 API

1. 访问 https://huggingface.co/settings/tokens
2. 创建免费的 API token
3. 设置环境变量:

```bash
# Windows PowerShell
$env:HUGGINGFACEHUB_API_TOKEN="your_token_here"

# Linux/Mac
export HUGGINGFACEHUB_API_TOKEN="your_token_here"
```

#### 选项 B: 使用本地 LLM (Ollama)

```bash
# 安装 Ollama
# https://ollama.ai

# 运行模型
ollama run llama2

# 设置环境变量
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
```

#### 选项 C: 使用模拟 LLM (仅演示)

如果不设置任何 API，将使用内置的模拟 LLM，适合快速测试工作流结构。

## 使用方法

### 基本使用

```python
from afl_workflow_langchain import AFLWorkflowLangChain

# 创建工作流实例
workflow = AFLWorkflowLangChain(use_free_api=True)

# 执行完整工作流
results = workflow.execute_full_workflow("vul_bn_exp.o")

# 查看结果
for stage, result in results.items():
    print(f"{stage}: {result}")
```

### 执行单个阶段

```python
from afl_workflow_langchain import AFLWorkflowLangChain, WorkflowStage

workflow = AFLWorkflowLangChain(use_free_api=True)

# 只执行分析阶段
result = workflow.execute_stage(
    WorkflowStage.ANALYZE,
    target_file="vul_bn_exp.o"
)
```

### 运行测试脚本

```bash
python test_workflow.py
```

## 代码结构

```
afl_workflow_langchain.py  # 主工作流实现
test_workflow.py           # 测试脚本
requirements.txt           # 依赖列表
README_LANGCHAIN.md       # 本文档
```

## 工作流实现细节

### LangChain 组件使用

1. **LLMChain**: 用于每个阶段的 LLM 调用
2. **SequentialChain**: 连接多个阶段
3. **Tools**: 实现具体的操作（文件分析、代码生成等）
4. **Memory**: 维护工作流状态
5. **Callbacks**: 监控工作流执行

### 状态管理

工作流使用 `WorkflowState` 数据类管理状态:

```python
@dataclass
class WorkflowState:
    stage: WorkflowStage
    target_file: Optional[str]
    functions_list: List[Dict]
    undefined_symbols: List[str]
    harness_code: Optional[str]
    # ... 更多状态字段
```

### 工具系统

每个工作流阶段对应一个或多个工具:

- `analyze_target_file`: 分析目标文件
- `generate_harness_code`: 生成 harness 代码
- `prepare_docker_env`: 准备 Docker 环境
- `compile_with_afl`: 编译程序
- `create_seed_files`: 创建种子文件
- `run_functional_test`: 运行功能测试
- `check_coverage`: 检查覆盖率
- `start_fuzzing`: 启动模糊测试
- `analyze_results`: 分析结果

## 扩展和自定义

### 添加新的工具

```python
Tool(
    name="your_tool",
    func=self._your_tool_function,
    description="工具描述"
)
```

### 自定义 LLM

```python
from langchain_openai import ChatOpenAI

# 使用自定义 LLM
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
workflow = AFLWorkflowLangChain(llm=llm, use_free_api=False)
```

### 添加新的工作流阶段

1. 在 `WorkflowStage` 枚举中添加新阶段
2. 在 `execute_stage` 方法中添加处理逻辑
3. 创建对应的工具函数

## 免费 API 选项

### 1. Hugging Face Hub (推荐)

- **免费额度**: 每天 1000 次请求
- **模型**: 支持多种开源模型
- **设置**: 需要 API token

### 2. LocalAI / Ollama

- **完全免费**: 本地运行
- **模型**: 支持多种开源模型
- **设置**: 需要本地安装

### 3. 模拟 LLM (演示用)

- **无需配置**: 直接使用
- **功能**: 仅返回模拟响应
- **用途**: 测试工作流结构

## 注意事项

1. **API 限制**: 免费 API 通常有速率限制
2. **模型大小**: 某些模型需要大量内存
3. **网络连接**: Hugging Face API 需要网络连接
4. **错误处理**: 工作流包含错误处理，但某些操作可能失败

## 故障排除

### 问题: Hugging Face API 初始化失败

**解决方案**:
- 检查 API token 是否正确设置
- 确认网络连接正常
- 尝试使用模拟 LLM

### 问题: 导入错误

**解决方案**:
```bash
pip install --upgrade langchain langchain-community
```

### 问题: 内存不足

**解决方案**:
- 使用较小的模型
- 使用 API 而不是本地模型
- 减少批处理大小

## 示例输出

```
============================================================
开始执行 AFL++ 完整工作流
============================================================

============================================================
执行阶段: analyze
============================================================
[工具] 分析目标文件: vul_bn_exp.o

============================================================
执行阶段: generate_harness
============================================================
[工具] 生成 harness 代码

...

============================================================
工作流执行完成
============================================================
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request!

## 相关资源

- [LangChain 文档](https://python.langchain.com/)
- [AFL++ 文档](https://github.com/AFLplusplus/AFLplusplus)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)

