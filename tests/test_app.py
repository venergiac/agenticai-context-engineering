import unittest
from unittest.mock import patch

import app


class AppTests(unittest.TestCase):
    def test_configure_langfuse_skips_client_when_tracing_disabled(self):
        with patch.dict("os.environ", {"LANGFUSE_TRACING_ENABLED": "false"}, clear=False):
            with patch("app.get_client") as mocked_get_client:
                app.configure_langfuse()

        mocked_get_client.assert_not_called()

    def test_build_graph_uses_selected_model_for_ollama_calls(self):
        graph = app.build_graph(model="demo-model")

        with patch("app.call_ollama", return_value="ok") as mocked:
            result = graph.invoke({"user_prompt": "hello"}, config={"configurable": {"model": "demo-model"}})

        self.assertEqual(result["direct_response"], "ok")
        self.assertEqual(result["compressed_response"], "ok")
        self.assertEqual(mocked.call_count, 2)
        self.assertEqual(mocked.call_args_list[0].kwargs["model"], "demo-model")
        self.assertEqual(mocked.call_args_list[1].kwargs["model"], "demo-model")


if __name__ == "__main__":
    unittest.main()
