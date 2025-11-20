"""
Agent Fuzz 使用示例

演示如何使用 AFL++ LangChain 工作流进行模糊测试自动化
"""

from afl_workflow_langchain import AFLWorkflowLangChain, WorkflowStage


def example_basic_usage():
    """基本使用示例"""
    print("="*60)
    print("示例 1: 基本使用 - 执行完整工作流")
    print("="*60)
    
    # 创建工作流实例
    workflow = AFLWorkflowLangChain(use_free_api=True)
    
    # 执行完整工作流
    results = workflow.execute_full_workflow("vul_bn_exp.o")
    
    # 打印结果摘要
    print("\n工作流执行结果:")
    for stage, result in results.items():
        print(f"  - {stage}: 完成")


def example_single_stage():
    """单个阶段执行示例"""
    print("\n" + "="*60)
    print("示例 2: 单个阶段执行")
    print("="*60)
    
    workflow = AFLWorkflowLangChain(use_free_api=True)
    
    # 只执行分析阶段
    result = workflow.execute_stage(
        WorkflowStage.ANALYZE,
        target_file="vul_bn_exp.o"
    )
    
    print(f"\n分析结果: {result.get('result', 'N/A')[:100]}...")


def example_custom_llm():
    """使用自定义 LLM 示例"""
    print("\n" + "="*60)
    print("示例 3: 使用自定义 LLM")
    print("="*60)
    
    # 注意: 这需要配置相应的 API
    # 示例: 使用 Hugging Face Hub
    try:
        from langchain_community.llms import HuggingFaceHub
        
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        
        workflow = AFLWorkflowLangChain(llm=llm, use_free_api=False)
        print("使用 Hugging Face Hub LLM")
        
    except ImportError:
        print("HuggingFaceHub 不可用，使用默认 LLM")
        workflow = AFLWorkflowLangChain(use_free_api=True)
    
    # 执行工作流
    results = workflow.execute_full_workflow("vul_bn_exp.o")
    print("工作流执行完成")


def example_step_by_step():
    """逐步执行示例"""
    print("\n" + "="*60)
    print("示例 4: 逐步执行工作流")
    print("="*60)
    
    workflow = AFLWorkflowLangChain(use_free_api=True)
    
    # 步骤 1: 分析目标文件
    print("\n步骤 1: 分析目标文件...")
    result1 = workflow.execute_stage(
        WorkflowStage.ANALYZE,
        target_file="vul_bn_exp.o"
    )
    print("✓ 分析完成")
    
    # 步骤 2: 生成 harness
    print("\n步骤 2: 生成 harness 代码...")
    result2 = workflow.execute_stage(WorkflowStage.GENERATE_HARNESS)
    print("✓ Harness 生成完成")
    
    # 步骤 3: 准备环境
    print("\n步骤 3: 准备 Docker 环境...")
    result3 = workflow.execute_stage(
        WorkflowStage.PREPARE_ENV,
        container_name="aflpp"
    )
    print("✓ 环境准备完成")
    
    # 可以继续执行其他阶段...
    print("\n工作流可以继续执行其他阶段...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Agent Fuzz - 使用示例")
    print("="*60)
    
    # 运行示例
    try:
        example_basic_usage()
        example_single_stage()
        example_step_by_step()
        
        print("\n" + "="*60)
        print("所有示例执行完成!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[错误] 示例执行失败: {e}")
        import traceback
        traceback.print_exc()

