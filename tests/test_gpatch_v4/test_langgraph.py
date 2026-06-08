import unittest


class LanggraphTest(unittest.TestCase):

    def test_langgraph_basic(self):
        # 本测试主要发现：
        # langgraph 使用上和 python 的 control flow 本质功能类似，不必担心。
        from typing import TypedDict

        try:
            import langgraph
            from langgraph.graph import StateGraph, END
        except ImportError:
            self.skipTest("no langgraph installed")

        class MessageState(TypedDict):
            nums: list
            last: int

        def plus_one(state: MessageState):
            nums = list(state["nums"])
            val = state["last"] + 1
            nums.append(val)
            return {"nums": nums, "last": val}

        def should_continue(state: MessageState):
            return "stop" if state["last"] > 10 else "continue"

        builder = StateGraph(MessageState)
        builder.add_node("plus_one", plus_one)
        builder.set_entry_point("plus_one")
        builder.add_conditional_edges(
            "plus_one",
            should_continue,
            {
                "continue": "plus_one",
                "stop": END
            },
        )

        graph = builder.compile()

        result = graph.invoke(
            {
                "nums": [],
                "last": 0
            },
            config={"recursion_limit": 50},
        )

        nums = result["nums"]
        self.assertEqual(nums, list(range(1, 12)))
