#!/usr/bin/env python3
"""检查数据同步状态"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.database.connection import get_supabase_client
from datetime import datetime

def main():
    print("=" * 60)
    print("A股数据同步状态检查")
    print("=" * 60)

    try:
        supabase = get_supabase_client()

        # 1. 股票基础信息
        result = supabase.table('stocks_info').select('*', count='exact').execute()
        total_stocks = len(result.data)
        print(f"\n📊 股票基础信息: {total_stocks} 只")

        # 2. 日线数据统计
        result = supabase.table('stock_daily').select('ts_code', count='exact').execute()
        all_records = result.data

        unique_stocks = set([r['ts_code'] for r in all_records])
        print(f"📈 日线数据: {len(all_records)} 条记录")
        print(f"📈 已同步股票: {len(unique_stocks)} 只")

        # 3. 日期范围
        if all_records:
            dates = [r.get('trade_date') for r in all_records if r.get('trade_date')]
            if dates:
                print(f"📅 日期范围: {min(dates)} 至 {max(dates)}")

        # 4. 同步状态文件
        state_file = Path(__file__).parent / ".sync_state.json"
        if state_file.exists():
            import json
            with open(state_file) as f:
                state = json.load(f)
            print(f"\n📋 同步状态文件:")
            print(f"   已完成: {len(state.get('completed', []))} 只")
            print(f"   失败: {len(state.get('failed', []))} 只")
            if state.get('last_sync'):
                print(f"   最后同步: {state['last_sync']}")
        else:
            print(f"\n⚠️  未找到同步状态文件 (.sync_state.json)")
            print("   建议运行: python scripts/sync_all_historical.py")

        # 5. 显示已同步的股票列表（前20只）
        if unique_stocks:
            print(f"\n📋 已同步股票示例（前20只）:")
            for code in sorted(list(unique_stocks))[:20]:
                count = len([r for r in all_records if r['ts_code'] == code])
                print(f"   {code}: {count} 条")

        print("\n" + "=" * 60)

    except Exception as e:
        print(f"❌ 查询失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
