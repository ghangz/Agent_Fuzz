"""
测试 AFL++ 工作流 - 使用免费 API
"""

import os
import sys
from afl_workflow_langchain import AFLWorkflowLangChain, WorkflowStage

# 尝试使用免费的 API
def test_with_free_api():
    """使用免费 API 测试工作流"""
    print("="*60)
    print("AFL++ 工作流测试 - 使用免费 API")
    print("="*60)
    
    # 检查是否有 Hugging Face API Token
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if hf_token:
        print(f"\n[信息] 检测到 Hugging Face API Token")
        print("[信息] 将使用 Hugging Face Hub API")
    else:
        print("\n[信息] 未检测到 HUGGINGFACEHUB_API_TOKEN")
        print("[信息] 将使用模拟 LLM (仅用于演示)")
        print("[提示] 要使用真实 API，请设置环境变量:")
        print("       export HUGGINGFACEHUB_API_TOKEN=your_token_here")
        print("       或访问 https://huggingface.co/settings/tokens 获取免费 token")
    
    try:
        # 创建工作流实例
        print("\n[初始化] 创建工作流实例...")
        workflow = AFLWorkflowLangChain(use_free_api=True)
        
        # 测试单个阶段
        print("\n" + "="*60)
        print("测试 1: 单个阶段执行 - 目标文件分析")
        print("="*60)
        
        result1 = workflow.execute_stage(
            WorkflowStage.ANALYZE,
            target_file="vul_bn_exp.o"
        )
        print(f"\n结果: {result1.get('result', 'N/A')[:200]}...")
        
        # 测试完整工作流
        print("\n" + "="*60)
        print("测试 2: 完整工作流执行")
        print("="*60)
        
        results = workflow.execute_full_workflow("vul_bn_exp.o")
        
        # 打印摘要
        print("\n" + "="*60)
        print("工作流执行摘要")
        print("="*60)
        
        for stage_name, stage_result in results.items():
            print(f"\n[{stage_name.upper()}]")
            if "result" in stage_result:
                result_text = stage_result["result"]
                # 只显示前 150 个字符
                if len(result_text) > 150:
                    print(result_text[:150] + "...")
                else:
                    print(result_text)
            elif "error" in stage_result:
                print(f"错误: {stage_result['error']}")
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_openai_compatible_api():
    """使用 OpenAI 兼容的免费 API 测试 (如 LocalAI, Ollama 等)"""
    print("\n" + "="*60)
    print("测试: 使用 OpenAI 兼容 API")
    print("="*60)
    
    # 检查是否有 OpenAI 兼容的 API
    api_base = os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_base or api_key:
        try:
            from langchain_openai import ChatOpenAI
            
            # 如果设置了 OPENAI_API_BASE，可能是本地 API
            if api_base:
                llm = ChatOpenAI(
                    base_url=api_base,
                    api_key=api_key or "not-needed",
                    model_name="gpt-3.5-turbo"
                )
            else:
                # 使用标准的 OpenAI API (需要付费)
                llm = ChatOpenAI(temperature=0.7)
            
            print(f"[信息] 使用 OpenAI 兼容 API: {api_base or 'default'}")
            
            workflow = AFLWorkflowLangChain(llm=llm, use_free_api=False)
            results = workflow.execute_full_workflow("vul_bn_exp.o")
            
            print("\n[成功] 使用 OpenAI 兼容 API 测试完成")
            return True
            
        except Exception as e:
            print(f"[错误] OpenAI 兼容 API 测试失败: {e}")
            return False
    else:
        print("[跳过] 未配置 OpenAI 兼容 API")
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("AFL++ LangChain 工作流测试")
    print("="*60)
    
    # 测试 1: 使用免费 API (Hugging Face 或模拟)
    success1 = test_with_free_api()
    
    # 测试 2: 尝试使用 OpenAI 兼容的 API
    success2 = test_with_openai_compatible_api()
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"免费 API 测试: {'✓ 成功' if success1 else '✗ 失败'}")
    print(f"OpenAI 兼容 API 测试: {'✓ 成功' if success2 else '✗ 跳过/失败'}")
    
    if success1 or success2:
        print("\n[成功] 至少一个测试通过!")
    else:
        print("\n[警告] 所有测试都未通过，请检查配置")


if __name__ == "__main__":
    main()

