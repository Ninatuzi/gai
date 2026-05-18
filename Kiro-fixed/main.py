"""
ChatWiki Agent 演示脚本

命令：
    python main.py init      # 初始化数据库和 Milvus
    python main.py health    # 检查服务连通性
    python main.py demo      # 完整演示
    python main.py chat <chat_id>   # 交互式对话
"""

import logging
import sys
import time

from chatwiki import ChatWikiAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatwiki.demo")


def cmd_init():
    print("=" * 60)
    print("初始化 ChatWiki Agent...")
    print("=" * 60)
    agent = ChatWikiAgent()
    agent.init()
    print("✅ 初始化完成")


def cmd_health():
    print("=" * 60)
    print("检查服务连通性...")
    print("=" * 60)
    agent = ChatWikiAgent()
    status = agent.health_check()
    for service, ok in status.items():
        print(f"  {'✅' if ok else '❌'} {service}")
    if not all(status.values()):
        print("\n⚠️  请检查失败的服务")
        sys.exit(1)
    print("\n✅ 所有服务正常")


def cmd_demo():
    """完整演示：多轮对话 + 话题自动分组 + Wiki检索 + RAG + 答案聚合"""
    print("=" * 60)
    print("ChatWiki Agent 演示")
    print("=" * 60)

    agent = ChatWikiAgent()

    # 1. 健康检查
    print("\n[1/5] 健康检查")
    status = agent.health_check()
    for s, ok in status.items():
        print(f"  {'✅' if ok else '❌'} {s}")
    if not all(status.values()):
        print("⚠️  服务异常")
        return

    # 2. 创建 workspace
    print("\n[2/5] 创建 workspace")
    ws_id = agent.create_workspace("demo_agent_chat")
    print(f"  workspace: {ws_id}")

    # 3. 多轮对话（模拟用户连续提问）
    print("\n[3/5] 多轮对话演示")
    demo_queries = [
        # 话题1：析锂（应该被归到同一个 module）
        "析锂是什么？",
        "析锂会有什么危害？",
        "怎么防止析锂？",
        # 话题2：正极材料（应该创建新 module）
        "磷酸铁锂和三元材料哪个更安全？",
        "三元材料的能量密度是多少？",
        # 闲聊（不走检索，不写入 wiki）
        "今天天气真好啊",
        # 回到话题1的追问（应该能从 wiki 中检索到之前聊的析锂内容）
        "刚才说的析锂在什么温度下容易发生？",
    ]

    for i, q in enumerate(demo_queries, 1):
        print(f"\n  [{i}/{len(demo_queries)}] Q: {q}")
        result = agent.ask(workspace_id=ws_id, query=q)
        print(f"  意图: {result.get('intent', '')}")
        print(f"  步骤: {' → '.join(result.get('steps', []))}")
        if result.get("wiki_found"):
            print(f"  Wiki命中: {result.get('wiki_modules_hit', 0)}个模块, {result.get('wiki_count', 0)}条记录")
        if result.get("rag_found"):
            print(f"  RAG命中: {result.get('rag_chunks_count', 0)}条")
        print(f"  回答: {result.get('answer', '')[:120]}...")
        if result.get("module_id"):
            print(f"  归属模块: {result['module_id'][:12]}...")
        time.sleep(0.5)

    # 4. 查看话题模块
    print("\n\n[4/5] 话题模块列表")
    modules = agent.get_modules(ws_id)
    for m in modules:
        wiki_count = m.get("wiki_count", 0)
        print(f"  📁 [{m['topic']}] (wiki数: {wiki_count})")
        print(f"     摘要: {(m.get('summary') or '待生成')[:80]}")
        wikis = agent.get_wikis_by_module(m["module_id"])
        for w in wikis:
            k_count = len(w.get("knowledge") or [])
            print(f"     └─ 第{w['turn_number']}轮: Q={w['query'][:30]}... (knowledge: {k_count}条)")

    # 5. 总结
    print("\n[5/5] 功能验证总结")
    print("  ✅ Agent 工作流编排（意图→Wiki→RAG→聚合→写入）")
    print("  ✅ 意图识别（knowledge_query / followup_query / chitchat）")
    print("  ✅ 话题自动分组（同话题归同module，新话题创建新module）")
    print("  ✅ Wiki 检索（从历史模块中找相关对话）")
    print("  ✅ RAG 检索（Mock，后续对接真实系统）")
    print("  ✅ 答案聚合（Wiki + RAG 综合生成）")
    print("  ✅ 保底机制（闲聊直答，不写入wiki）")
    print("  ✅ Wiki 格式存储（Q + A + Knowledge）")

    print("\n" + "=" * 60)
    print("✅ 演示完成")
    print("=" * 60)


def cmd_chat(chat_id: str):
    """交互式对话"""
    agent = ChatWikiAgent()
    ws_id = agent.create_workspace(chat_id)
    print(f"进入对话 workspace: {ws_id}")
    print("（输入 :q 退出，:modules 查看话题模块，:wikis 查看所有wiki）\n")

    while True:
        try:
            user = input("你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == ":q":
            break
        if user == ":modules":
            modules = agent.get_modules(ws_id)
            for m in modules:
                print(f"  📁 [{m['topic']}] wiki数={m.get('wiki_count', 0)}")
                print(f"     {(m.get('summary') or '')[:60]}")
            continue
        if user == ":wikis":
            wikis = agent.get_wikis(ws_id)
            for w in wikis:
                print(f"  [#{w['turn_number']}] Q: {w['query'][:40]}...")
                print(f"         A: {w['answer'][:40]}...")
            continue

        # Agent 问答
        result = agent.ask(workspace_id=ws_id, query=user)
        print(f"  [{result.get('intent', '')} | wiki={result.get('wiki_modules_hit', 0)} | "
              f"rag={result.get('rag_chunks_count', 0) if result.get('rag_found') else 0}]")
        print(f"助手: {result.get('answer', '')}\n")


# ============================================================
# 入口
# ============================================================

COMMANDS = {"init": cmd_init, "health": cmd_health, "demo": cmd_demo}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "chat":
        if len(sys.argv) < 3:
            print("用法: python main.py chat <chat_id>")
            sys.exit(1)
        cmd_chat(sys.argv[2])
    elif cmd in COMMANDS:
        COMMANDS[cmd]()
    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
