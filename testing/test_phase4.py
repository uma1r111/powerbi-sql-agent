import asyncio
import sys
import os
import pandas as pd
from tabulate import tabulate


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.graph import agent


async def test_agent():
    # Pandas display settings → no truncation
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    
    # Run query
    result = await agent.process_query("Show customer from Germany")

    print("\n=== FINAL RESULT ===")
    print(f"✅ Success: {result.processing_complete}")
    print(f"🗂️ Selected tables: {result.selected_tables}")
    print(f"📝 SQL Query Used:\n{result.cleaned_sql}")
    print(f"📊 Rows Returned: {result.result_count}")
    print(f"⏱️ Execution Time: {getattr(result, 'execution_time', 'N/A')} seconds")

    # Format query results
    if hasattr(result, "execution_result") and result.execution_result:
        df = pd.DataFrame(result.execution_result)

        print("\n=== Query Results ===")
        print(tabulate(df, headers="keys", tablefmt="psql", showindex=False))
    else:
        print("\n⚠️ No rows returned.\n")

    # Show conversation
    print("\n=== Conversation History ===")
    for msg in result.messages:
        role = "👤 Human" if msg.type == "human" else "🤖 AI"
        print(f"{role}: {msg.content}")


# Run the test
if __name__ == "__main__":
    asyncio.run(test_agent())
