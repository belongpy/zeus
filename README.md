# ⚡ Zeus - Standalone Wallet Analysis System

**Zeus** is a sophisticated wallet analysis system designed for automated trading decisions. It provides binary YES/NO decisions on whether to follow wallets and their trading strategies, specifically designed for bot integration.

## 🎯 Key Features

### Binary Decision System
- **Follow Wallet**: YES/NO decision based on composite score ≥65
- **Follow Sells**: YES/NO decision based on exit quality ≥70%
- **Bot-Ready Output**: Direct integration with trading bots

### Advanced Analysis
- **30-Day Analysis Window** (vs Phoenix's 7-day)
- **Smart Token Sampling**: 5 initial → 10 if inconclusive  
- **Volume Qualifier**: Minimum 6 unique token trades
- **New Scoring System**: 5-component weighted analysis

### Professional Output
- **CSV Export** with bot configuration
- **TP/SL Strategy Matrix** with custom recommendations
- **Performance Scoring** with detailed breakdown

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download Zeus system
cd zeus/

# Install dependencies
pip install -r requirements.txt

# Make CLI executable (Linux/Mac)
chmod +x zeus_cli.py
```

### 2. Configuration

```bash
# Configure required API key (Cielo Finance)
python zeus_cli.py configure --cielo-api-key YOUR_CIELO_KEY

# Configure optional APIs for enhanced analysis
python zeus_cli.py configure --birdeye-api-key YOUR_BIRDEYE_KEY
python zeus_cli.py configure --helius-api-key YOUR_HELIUS_KEY
```

### 3. Add Wallets

Create `wallets.txt` with wallet addresses (one per line):

```
7xG8k9mPqR3nW2sJ5tY8vL4hE6dF1aZ9bN3cM7uV2iK1
9aB2c3D4e5F6g7H8i9J1k2L3m4N5o6P7q8R9s1T2u3V4
5eF7g8H9i1J2k3L4m5N6o7P8q9R1s2T3u4V5w6X7y8Z9
```

### 4. Run Analysis

```bash
# Interactive menu
python zeus_cli.py

# Command line analysis
python zeus_cli.py analyze --wallets wallets.txt
python zeus_cli.py analyze --wallet 7xG8k9mPqR3nW2sJ5tY8vL4hE6dF1aZ9bN3cM7uV2iK1
```

---

## 📊 Scoring System

Zeus uses a **5-component weighted scoring system** (0-100 scale):

### Component Weights
1. **Risk-Adjusted Performance (30%)**
   - Median ROI, Standard Deviation, Volume Qualifier
2. **Distribution Quality (25%)**  
   - Moonshot rate, Big win distribution, Loss management
3. **Trading Discipline (20%)**
   - Loss cutting, Exit behavior, Flipper detection
4. **Market Impact Awareness (15%)**
   - Bet sizing, Market cap strategy
5. **Consistency & Reliability (10%)**
   - Activity pattern, Red flag detection

### Volume Qualifier
- **≥6 tokens**: 100 points (Baseline - no penalties)
- **4-5 tokens**: 80 points (Emerging wallet)  
- **2-3 tokens**: 60 points (Very new wallet)
- **<2 tokens**: Disqualified

---

## 🎯 Binary Decision Matrix

### Decision 1: Follow Wallet?
| Composite Score | Min Tokens | Bot Behavior | Flipper Score | Decision |
|----------------|------------|--------------|---------------|----------|
| ≥65 | ≥6 | <20% same-block | <40 flipper | ✅ **YES** |
| <65 | Any | Any | Any | ❌ **NO** |
| Any | <6 | Any | Any | ❌ **NO** |
| Any | Any | >20% same-block | Any | ❌ **NO** |

### Decision 2: Follow Sells?
| Exit Quality | Peak Capture | Dump Rate | Profit Consistency | Early Exits | Decision |
|-------------|--------------|-----------|-------------------|-------------|----------|
| ≥70% | ≥60% | ≤25% | ≥60% | ≤30% | ✅ **YES** |
| All other combinations | | | | | ❌ **NO** |

---

## 📋 TP/SL Strategy Matrix

### Mirror Strategy (Follow Sells = YES)
- **Copy their exits** with safety buffer
- TP1: 80% of their average exit
- TP2: 150% of their average exit  
- TP3: 300% of their average exit
- Stop Loss: -35%

### Custom Strategy (Follow Sells = NO)
| Pattern | TP1 | TP2 | TP3 | Stop Loss | Reasoning |
|---------|-----|-----|-----|-----------|-----------|
| **Gem Hunter** | 100% | 300% | 800% | -40% | Finds gems but exits poorly |
| **Consistent Scalper** | 50% | 100% | 200% | -25% | Consistent but exits early |
| **Volatile Trader** | 60% | 150% | 400% | -30% | Account for volatility |
| **Mixed Strategy** | 75% | 200% | 500% | -35% | Balanced approach |

---

## 🖥️ CLI Commands

### Configuration
```bash
zeus configure --cielo-api-key YOUR_KEY      # Required
zeus configure --birdeye-api-key YOUR_KEY    # Recommended  
zeus configure --helius-api-key YOUR_KEY     # Optional
zeus configure --rpc-url YOUR_RPC_URL        # Optional
```

### Analysis
```bash
zeus analyze --wallets wallets.txt                    # Batch analysis
zeus analyze --wallet ADDRESS --output results.csv   # Single wallet
zeus analyze --wallets file.txt --days 30            # Custom timeframe
zeus status                                           # System status
```

### Interactive Menu
```bash
zeus                                         # Launch interactive menu
```

---

## 📁 File Structure

```
zeus/
├── zeus_cli.py              # Main CLI entry point
├── zeus_analyzer.py         # Core analysis engine  
├── zeus_scorer.py           # Scoring system
├── zeus_api_manager.py      # API integration
├── zeus_export.py           # CSV export
├── zeus_config.py           # Configuration management
├── zeus_utils.py            # Utility functions
├── requirements.txt         # Dependencies
├── README.md               # This file
├── config/
│   └── zeus_config.json    # Default configuration
├── outputs/                # Analysis results  
└── wallets.txt            # Wallet addresses
```

---

## 🔧 Configuration

Zeus supports multiple configuration methods:

### 1. Configuration File (`~/.zeus_config.json`)
```json
{
  "api_keys": {
    "cielo_api_key": "your_key_here",
    "birdeye_api_key": "your_key_here",
    "helius_api_key": "your_key_here"
  },
  "analysis": {
    "days_to_analyze": 30,
    "min_unique_tokens": 6,
    "composite_score_threshold": 65.0,
    "exit_quality_threshold": 70.0
  }
}
```

### 2. Environment Variables
```bash
export ZEUS_CIELO_API_KEY="your_key"
export ZEUS_BIRDEYE_API_KEY="your_key"  
export ZEUS_HELIUS_API_KEY="your_key"
export ZEUS_ANALYSIS_DAYS=30
export ZEUS_MIN_TOKENS=6
```

### 3. Command Line Arguments
```bash
zeus configure --cielo-api-key YOUR_KEY --days 30
```

---

## 📈 Output Format

### CSV Export Columns
- **Basic Info**: `rank`, `wallet_address`, `analysis_timestamp`
- **Binary Decisions**: `follow_wallet`, `follow_sells` 
- **Bot Config**: `copy_entries`, `copy_exits`, `tp1_percent`, `tp2_percent`, `tp3_percent`, `stop_loss_percent`
- **Scoring**: `composite_score`, `risk_adjusted_score`, `distribution_score`, etc.
- **Volume**: `unique_tokens_traded`, `tokens_analyzed`, `volume_qualifier_tier`

### Example Output
```csv
rank,wallet_address,follow_wallet,follow_sells,tp1_percent,tp2_percent,tp3_percent,stop_loss_percent,composite_score
1,7xG8k9m...2iK1,YES,NO,100,300,800,-40,78.5
2,9aB2c3D...3V4,YES,YES,75,150,300,-35,72.1
3,5eF7g8H...8Z9,NO,NO,0,0,0,-35,58.3
```

---

## 🔌 API Requirements

### Required APIs
- **Cielo Finance API** - Wallet trading statistics (REQUIRED)

### Recommended APIs  
- **Birdeye API** - Token price data and market info
- **Helius API** - Enhanced transaction parsing

### Fallback APIs
- **Solana RPC** - Direct blockchain access (always available)

---

## ⚙️ Advanced Features

### Smart Token Sampling
1. **Initial Analysis**: 5 most recent token trades
2. **Conclusive Check**: Clear win/loss pattern detected?
3. **Extended Analysis**: If inconclusive, expand to 10 tokens
4. **Final Decision**: Generate binary decisions with confidence

### Volume Qualifier System
- **Baseline Tier** (≥6 tokens): Full scoring, no penalties
- **Emerging Tier** (4-5 tokens): 80% score multiplier
- **New Tier** (2-3 tokens): 60% score multiplier  
- **Disqualified** (<2 tokens): Cannot analyze

### Pattern Recognition
- **Gem Hunter**: High variance, high upside potential
- **Consistent Scalper**: Low variance, steady profits
- **Volatile Trader**: High variance trading style
- **Position Trader**: Longer hold times
- **Flipper**: Very quick trades (same-block detection)

---

## 🚨 Error Handling

Zeus handles various error conditions gracefully:

- **Insufficient Volume**: `<6 unique tokens in 30 days`
- **API Failures**: `Fallback to available APIs`  
- **Data Quality**: `Missing or corrupt transaction data`
- **Network Issues**: `Retry logic with exponential backoff`
- **Invalid Wallets**: `Address validation and error reporting`

---

## 🛠️ Troubleshooting

### Common Issues

**1. "Cielo Finance API not configured"**
```bash
zeus configure --cielo-api-key YOUR_KEY
```

**2. "No wallets found in wallets.txt"**
- Create `wallets.txt` file with valid Solana addresses
- One wallet address per line
- Lines starting with `#` are ignored (comments)

**3. "Insufficient unique tokens"**
- Wallet needs ≥6 unique token trades in 30 days
- Consider analyzing older/more active wallets

**4. Low success rate**
- Check API key validity with `zeus status`
- Verify wallet addresses are valid Solana addresses
- Check network connectivity

### Debug Mode
```bash
# Enable debug logging
export ZEUS_LOG_LEVEL=DEBUG
zeus analyze --wallets wallets.txt
```

### System Status
```bash
zeus status                    # Check API connectivity
zeus configure                 # Review current configuration
```

---

## 📊 Performance

### Analysis Speed
- **Single Wallet**: ~5-15 seconds
- **Batch (10 wallets)**: ~2-5 minutes  
- **Large Batch (100 wallets)**: ~20-50 minutes

### API Usage
- **Cielo Finance**: ~5-10 calls per wallet
- **Birdeye**: ~2-5 calls per wallet (optional)
- **Helius**: ~1-3 calls per wallet (optional)
- **RPC**: Fallback usage only

---

## 🔄 Updates & Migration

Zeus automatically handles configuration migration between versions. When upgrading:

1. **Backup** your current `~/.zeus_config.json`
2. **Install** new Zeus version
3. **Run** `zeus configure` to migrate settings
4. **Test** with `zeus status` to verify functionality

---

## 📝 License & Support

### License
Zeus is proprietary software. All rights reserved.

### Support
- **Documentation**: This README and inline help
- **CLI Help**: `zeus --help` or `zeus COMMAND --help`
- **System Status**: `zeus status` for diagnostics
- **Interactive Menu**: Run `zeus` without arguments

### Contributing
Zeus is a standalone system extracted from Phoenix. For feature requests or bug reports, use the standard issue tracking process.

---

## 🎉 Getting Started Checklist

- [ ] Install Python dependencies (`pip install -r requirements.txt`)
- [ ] Configure Cielo Finance API key (REQUIRED)
- [ ] Configure Birdeye API key (RECOMMENDED)  
- [ ] Create `wallets.txt` with target wallet addresses
- [ ] Run test analysis (`zeus analyze --wallets wallets.txt`)
- [ ] Check output CSV in `outputs/` directory
- [ ] Configure your trading bot with Zeus binary decisions

**Ready to analyze wallets with Zeus!** ⚡

---

*Zeus - Binary Decisions for Automated Trading*