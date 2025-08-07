from typing import List

class KnowledgeBaseBuilder:
    """
    A minimal, self-contained knowledge base builder for diagnostics.
    It has no external dependencies.
    """
    def __init__(self):
        pass

    def get_predicates_from_text(self, article_text: str) -> List[str]:
        """
        A simplified function that returns a hardcoded predicate for the demo
        if it sees the relevant text.
        """
        if "human oversight" in article_text.lower():
            return ["requires(system, component='human_oversight')"]
        return []
