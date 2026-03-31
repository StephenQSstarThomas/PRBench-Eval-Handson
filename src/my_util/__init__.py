import re
from typing import Dict


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    """Parse XML-style tags from a string.

    Extracts tags in the format <tag_name>content</tag_name> and returns
    a dict mapping tag names to their stripped content.
    """
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}
