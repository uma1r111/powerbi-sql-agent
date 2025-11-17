import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flow.graph import agent
from tools.error_manager import error_manager


def print_banner():
    """Print welcome banner"""
    print("=" * 70)
    print("🤖 PowerBI SQL Agent - Conversational Interface")
    print("=" * 70)
    print("\n📊 Ask questions about the Northwind database in natural language")
    print("\n💡 Special Commands:")
    print("  /detailed     - Switch to detailed table view")
    print("  /simple       - Switch to conversational view (default)")
    print("  /sql on       - Show SQL queries in responses")
    print("  /sql off      - Hide SQL queries")
    print("  /stats        - Show error statistics")
    print("  /debug on     - Show execution details")
    print("  /debug off    - Hide execution details")
    print("  /help         - Show this help")
    print("  /clear        - Clear conversation history")
    print("  exit/quit     - End session")
    print("=" * 70)
    print()

def print_help():
    """Print help information"""
    print("\n" + "=" * 70)
    print("📚 How to use:")
    print("=" * 70)
    print("\n🔹 Ask natural language questions:")
    print("  • How many customers do we have?")
    print("  • Show me customers from Germany")
    print("  • What's the total revenue by product category?")
    print("\n🔹 Follow-up questions:")
    print("  • How many orders did they place?")
    print("  • Show me their contact information")
    print("\n🔹 Toggle display modes:")
    print("  • /detailed - See full data tables")
    print("  • /simple - Get conversational summaries")
    print("  • /sql on - See the generated SQL")
    print("=" * 70)
    print()

def main():
    print_banner()
    
    session_id = "terminal_session_001"
    
    # Session state
    response_format = "conversational"
    show_sql = False
    show_debug = False
    
    # Track conversation for context
    conversation_count = 0

    # NEW: Maintain persistent state across queries in session
    current_state = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle exit commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye! Session ended.")
                print(f"📊 Total queries processed: {conversation_count}")
                break
            
            # Handle special commands
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/help':
                    print_help()
                    continue
                
                elif command == '/detailed':
                    response_format = "detailed"
                    print("✅ Switched to detailed table view")
                    continue
                
                elif command == '/simple':
                    response_format = "conversational"
                    print("✅ Switched to conversational view")
                    continue
                
                elif command == '/sql on':
                    show_sql = True
                    print("✅ SQL queries will be shown in responses")
                    continue
                
                elif command == '/sql off':
                    show_sql = False
                    print("✅ SQL queries will be hidden")
                    continue
                
                elif command == '/debug on':
                    show_debug = True
                    print("✅ Debug information enabled")
                    continue
                
                elif command == '/debug off':
                    show_debug = False
                    print("✅ Debug information disabled")
                    continue
                
                elif command == '/clear':
                    session_id = f"terminal_session_{conversation_count + 1}"
                    conversation_count = 0
                    print("✅ Conversation history cleared. Starting fresh session.")
                    continue

                elif command == '/stats':
                    stats = error_manager.get_error_stats()
                    print("\n📊 Error Statistics for this session:")
                    print(f"Total Errors: {stats['total_errors']}")
                    print(f"Resolved: {stats['resolved']} ({stats['resolution_rate']:.1f}%)")
                    print(f"Retryable: {stats['retryable']}")
                    print("\nBy Category:")
                    for category, count in stats['by_category'].items():
                        print(f"  {category}: {count}")
                    print("\nBy Severity:")
                    for severity, count in stats['by_severity'].items():
                        print(f"  {severity}: {count}")
                    continue
                
                else:
                    print(f"❌ Unknown command: {user_input}")
                    print("💡 Type /help to see available commands")
                    continue
            
            # Process query
            print("\n🔄 Processing", end="", flush=True)
            for _ in range(3):
                print(".", end="", flush=True)
                import time
                time.sleep(0.3)
            print()
            
            # CRITICAL FIX: Pass current_state to maintain context
            from flow.graph import agent
            result = agent.process_query_sync(
                user_input, 
                session_id=session_id,
                existing_state=current_state  # Pass previous state
            )
            
            # CRITICAL: Update current_state for next query
            current_state = result
            
            # Apply current preferences (they might have been reset)
            current_state.response_format = response_format
            current_state.show_sql = show_sql
            current_state.show_execution_details = show_debug
            
            # Display response
            if result.messages and len(result.messages) > 0:
                last_message = result.messages[-1]
                print(f"\n🤖 Agent:\n{last_message.content}\n")
            else:
                print("\n🤖 Agent: [No response generated]\n")
            
            # Debug info
            if show_debug:
                print(f"\n🔍 Debug Info:")
                print(f"   Follow-up: {result.is_follow_up_query}")
                print(f"   Tables: {result.selected_tables}")
                print(f"   Last Topic: {result.last_query_topic}")
                print(f"   Last Tables: {result.last_tables_used}")
                print(f"   Context Window: {len(result.context_window)} queries")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted.")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            if show_debug:
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    main()
