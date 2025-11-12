"""
Main entry point for the trading bot.
"""
import asyncio
from src.orchestrator import TradingBot

def main():
    """Run the trading bot."""
    print("=" * 70)
    print(" " * 15 + "Quant Trading Bot")
    print("=" * 70)
    print()
    print("Starting bot with configuration from config.yaml...")
    print()
    
    bot = TradingBot()
    
    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise


if __name__ == "__main__":
    main()
