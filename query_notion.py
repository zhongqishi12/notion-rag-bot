import os
import re

from notion_client import Client
from dotenv import load_dotenv

load_dotenv()

notion = Client(auth=os.environ["NOTION_TOKEN"])


def sanitize_filename(name: str) -> str:
    """清理非法文件名字符"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def query_notion_database(database_id, query_filter=None):
    """
    获取数据库下的所有 page，支持分页
    """
    all_pages = []
    cursor = None
    has_more = True

    while has_more:
        query = {"database_id": database_id, "page_size": 100}
        if query_filter:
            query["filter"] = query_filter
        if cursor:
            query["start_cursor"] = cursor

        response = notion.databases.query(**query)
        all_pages.extend(response.get("results", []))

        has_more = response.get("has_more", False)
        cursor = response.get("next_cursor")

    return all_pages


def get_page_content(page_id):
    """
    获取单个 page 的正文内容（blocks）
    """
    all_blocks = []
    cursor = None
    has_more = True

    while has_more:
        resp = notion.blocks.children.list(page_id, start_cursor=cursor)
        all_blocks.extend(resp.get("results", []))
        has_more = resp.get("has_more", False)
        cursor = resp.get("next_cursor")

    texts = []
    for block in all_blocks:
        btype = block["type"]
        if "rich_text" in block[btype]:
            texts.append("".join([t["plain_text"] for t in block[btype]["rich_text"]]))
    return "\n".join(texts)


def get_page_title(page):
    """尝试获取 page 的 title 属性"""
    properties = page.get("properties", {})
    for prop in properties.values():
        if prop["type"] == "title":
            rich = prop["title"]
            if rich:
                return rich[0]["plain_text"]
    return "Untitled"


if __name__ == "__main__":
    database_id = "fb0fa29a80364b139c98422638b866b6"  # 你的 Notion 数据库 ID
    pages = query_notion_database(database_id)

    print(f"共获取 {len(pages)} 条 page")

    os.makedirs("notion_md", exist_ok=True)

    for page in pages:
        page_id = page["id"]
        title = sanitize_filename(get_page_title(page))
        content = get_page_content(page_id)

        md_file = os.path.join("notion_md", f"{title}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n\n")
            f.write(content)
        print(f"✅ 已导出 {md_file}")
