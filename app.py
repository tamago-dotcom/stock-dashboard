import streamlit as st
import yfinance as yf
import mplfinance as mpf
from curl_cffi import requests as curl_requests

_session = curl_requests.Session(impersonate="chrome")
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib_fontja  # noqa: F401 — import するだけで日本語フォントが有効になる
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

matplotlib.use("Agg")
mpl.rcParams["axes.unicode_minus"] = False

FINANCIAL_TRANSLATIONS = {
    # ── 損益計算書 ──────────────────────────────
    "Total Revenue":                                    "売上高",
    "Cost Of Revenue":                                  "売上原価",
    "Gross Profit":                                     "売上総利益",
    "Operating Expense":                                "営業費用",
    "Selling General Administrative":                   "販売費及び一般管理費",
    "Selling And Marketing Expense":                    "販売促進費",
    "General And Administrative Expense":               "一般管理費",
    "Research And Development":                         "研究開発費",
    "Other Operating Expenses":                         "その他営業費用",
    "Total Expenses":                                   "費用合計",
    "Operating Income":                                 "営業利益",
    "EBIT":                                             "EBIT",
    "EBITDA":                                           "EBITDA",
    "Normalized EBITDA":                                "調整後EBITDA",
    "Reconciled Depreciation":                          "調整後減価償却費",
    "Reconciled Cost Of Revenue":                       "調整後売上原価",
    "Interest Income":                                  "受取利息",
    "Interest Expense":                                 "支払利息",
    "Net Interest Income":                              "純受取利息",
    "Other Income Expense":                             "その他収支",
    "Other Non Operating Income Expenses":              "その他営業外収支",
    "Special Income Charges":                           "特別損益",
    "Gain On Sale Of Security":                         "有価証券売却益",
    "Earnings From Equity Interest":                    "持分法投資損益",
    "Pretax Income":                                    "税引前当期純利益",
    "Income Before Tax":                                "税引前当期純利益",
    "Tax Provision":                                    "法人税等",
    "Income Tax Expense":                               "法人税費用",
    "Net Income":                                       "当期純利益",
    "Net Income Common Stockholders":                   "普通株主帰属純利益",
    "Net Income Including Noncontrolling Interests":    "非支配株主持分含む純利益",
    "Net Income From Continuing Operations":            "継続事業純利益",
    "Net Income From Continuing And Discontinued Operation": "継続・非継続事業純利益",
    "Minority Interests":                               "少数株主持分損益",
    "Normalized Income":                                "調整後純利益",
    "Diluted NI Available To Com Stockholders":         "希薄化後普通株主帰属純利益",
    "Total Unusual Items":                              "特別損益合計",
    "Total Unusual Items Excluding Goodwill":           "特別損益合計（のれん除く）",
    "Basic EPS":                                        "基本EPS",
    "Diluted EPS":                                      "希薄化後EPS",
    "Basic Average Shares":                             "基本平均株式数",
    "Diluted Average Shares":                           "希薄化後平均株式数",
    # 表記ゆれ・追加項目（損益計算書）
    "Operating Revenue":                                "営業収益",
    "Selling General And Administration":               "販売費及び一般管理費",
    "Net Non Operating Interest Income Expense":        "営業外純受取利息",
    "Interest Expense Non Operating":                   "営業外支払利息",
    "Interest Income Non Operating":                    "営業外受取利息",
    "Net Income From Continuing Operation Net Minority Interest": "少数株主持分控除後継続事業純利益",
    "Net Income Continuous Operations":                 "継続事業純利益",
    "Diluted NI Availto Com Stockholders":              "希薄化後普通株主帰属純利益",
    "Tax Rate For Calcs":                               "実効税率",
    "Tax Effect Of Unusual Items":                      "特別損益の税効果",
    "Total Operating Income As Reported":               "報告ベース営業利益合計",
    "Average Dilution Earnings":                        "希薄化調整後平均利益",
    "Total Other Finance Cost":                         "その他金融費用合計",
    "Otherunder Preferred Stock Dividend":              "優先株配当金控除後その他",
    "Other Special Charges":                            "その他特別費用",
    "Write Off":                                        "資産償却・評価損",
    "Depreciation And Amortization In Income Statement":"損益計算書上の減価償却費",
    "Depreciation Income Statement":                    "損益計算書上の減価償却費（単独）",
    # ── バランスシート ───────────────────────────
    "Total Assets":                                     "総資産",
    "Current Assets":                                   "流動資産合計",
    "Cash And Cash Equivalents":                        "現金及び現金同等物",
    "Cash Cash Equivalents And Short Term Investments": "現金・短期投資合計",
    "Other Short Term Investments":                     "その他短期投資",
    "Accounts Receivable":                              "売掛金",
    "Other Receivables":                                "その他売掛金",
    "Inventory":                                        "棚卸資産",
    "Other Current Assets":                             "その他流動資産",
    "Total Non Current Assets":                         "非流動資産合計",
    "Net PPE":                                          "有形固定資産（純額）",
    "Gross PPE":                                        "有形固定資産（総額）",
    "Accumulated Depreciation":                         "減価償却累計額",
    "Land And Improvements":                            "土地・改良費",
    "Buildings And Improvements":                       "建物・改良費",
    "Machinery Furniture Equipment":                    "機械・什器・設備",
    "Leases":                                           "リース資産",
    "Other Properties":                                 "その他固定資産",
    "Goodwill":                                         "のれん",
    "Intangible Assets":                                "無形固定資産",
    "Investments And Advances":                         "投資・前払金",
    "Long Term Equity Investment":                      "長期株式投資",
    "Other Non Current Assets":                         "その他非流動資産",
    "Total Liabilities Net Minority Interest":          "負債合計（少数株主持分除く）",
    "Current Liabilities":                              "流動負債合計",
    "Accounts Payable":                                 "買掛金",
    "Payables":                                         "支払債務",
    "Payables And Accrued Expenses":                    "支払債務・未払費用",
    "Current Debt":                                     "短期借入金",
    "Current Debt And Capital Lease Obligation":        "短期借入金・リース債務",
    "Current Deferred Revenue":                         "前受収益（流動）",
    "Current Provisions":                               "引当金（流動）",
    "Other Current Liabilities":                        "その他流動負債",
    "Total Non Current Liabilities Net Minority Interest": "非流動負債合計",
    "Long Term Debt":                                   "長期借入金",
    "Long Term Debt And Capital Lease Obligation":      "長期借入金・リース債務",
    "Capital Lease Obligations":                        "リース債務",
    "Long Term Provisions":                             "引当金（固定）",
    "Non Current Deferred Revenue":                     "前受収益（固定）",
    "Non Current Deferred Taxes Liabilities":           "繰延税金負債（固定）",
    "Other Non Current Liabilities":                    "その他非流動負債",
    "Total Equity Gross Minority Interest":             "純資産合計（少数株主持分含む）",
    "Stockholders Equity":                              "株主資本",
    "Common Stock Equity":                              "普通株主資本",
    "Common Stock":                                     "普通株式",
    "Retained Earnings":                                "利益剰余金",
    "Additional Paid In Capital":                       "資本剰余金",
    "Treasury Stock":                                   "自己株式",
    "Capital Stock":                                    "資本金",
    "Preferred Stock":                                  "優先株式",
    "Other Equity Adjustments":                         "その他資本調整",
    "Total Debt":                                       "有利子負債合計",
    "Net Debt":                                         "純有利子負債",
    "Working Capital":                                  "運転資本",
    "Invested Capital":                                 "投下資本",
    "Total Capitalization":                             "資本合計",
    "Net Tangible Assets":                              "純有形資産",
    "Share Issued":                                     "発行済株式数",
    "Ordinary Shares Number":                           "普通株式数",
    # 表記ゆれ・追加項目（バランスシート）
    "Treasury Shares Number":                           "自己株式数",
    "Tangible Book Value":                              "有形純資産",
    "Gains Losses Not Affecting Retained Earnings":     "利益剰余金に影響しない損益",
    "Tradeand Other Payables Non Current":              "非流動買掛金・その他支払債務",
    "Long Term Capital Lease Obligation":               "長期資本リース債務",
    "Current Capital Lease Obligation":                 "流動資本リース債務",
    "Current Deferred Liabilities":                     "流動繰延負債",
    "Other Current Borrowings":                         "その他流動借入金",
    "Commercial Paper":                                 "コマーシャルペーパー",
    "Current Accrued Expenses":                         "未払費用（流動）",
    "Total Tax Payable":                                "未払税金合計",
    "Income Tax Payable":                               "未払法人税",
    "Non Current Deferred Assets":                      "非流動繰延資産",
    "Non Current Deferred Taxes Assets":                "繰延税金資産（固定）",
    "Other Investments":                                "その他投資",
    "Investmentin Financial Assets":                    "金融資産投資",
    "Available For Sale Securities":                    "売却可能有価証券",
    "Properties":                                       "固定資産",
    "Receivables":                                      "受取債権",
    "Cash Equivalents":                                 "現金同等物",
    "Cash Financial":                                   "金融現金",
    "Minority Interest":                                "少数株主持分",
    "Other Equity Interest":                            "その他持分",
    "Employee Benefits":                                "従業員給付引当金",
    "Non Current Pension And Other Postretirement Benefit Plans": "退職給付・その他退職後給付（固定）",
    "Non Current Deferred Liabilities":                 "非流動繰延負債",
    "Other Payable":                                    "その他支払債務",
    "Non Current Accounts Receivable":                  "長期売掛金",
    "Investmentsin Joint Venturesat Cost":              "合弁会社への投資（取得原価）",
    "Investmentsin Associatesat Cost":                  "関連会社への投資（取得原価）",
    "Goodwill And Other Intangible Assets":             "のれん及びその他無形固定資産",
    "Other Intangible Assets":                          "その他無形固定資産",
    "Construction In Progress":                         "建設仮勘定",
    "Finished Goods":                                   "製品",
    "Work In Process":                                  "仕掛品",
    "Raw Materials":                                    "原材料",
    "Receivables Adjustments Allowances":               "貸倒引当金",
    "Taxes Receivable":                                 "未収税金",
    "Gross Accounts Receivable":                        "売掛金（総額）",
    # ── キャッシュフロー ─────────────────────────
    "Operating Cash Flow":                              "営業活動によるキャッシュフロー",
    "Net Income From Continuing Operations":            "継続事業純利益",
    "Depreciation And Amortization":                    "減価償却費・償却費",
    "Depreciation":                                     "減価償却費",
    "Amortization":                                     "償却費",
    "Stock Based Compensation":                         "株式報酬",
    "Deferred Tax":                                     "繰延税金",
    "Change In Working Capital":                        "運転資本の変動",
    "Change In Receivables":                            "売掛金の変動",
    "Changes In Account Receivables":                   "売掛金の変動",
    "Change In Inventory":                              "棚卸資産の変動",
    "Change In Payables":                               "買掛金の変動",
    "Change In Other Current Assets":                   "その他流動資産の変動",
    "Change In Other Current Liabilities":              "その他流動負債の変動",
    "Other Non Cash Items":                             "その他非現金項目",
    "Other Operating Activities":                       "その他営業活動",
    "Investing Cash Flow":                              "投資活動によるキャッシュフロー",
    "Capital Expenditure":                              "設備投資",
    "Net PPE Purchase And Sale":                        "有形固定資産の取得・売却",
    "Purchase Of Investment":                           "投資有価証券の取得",
    "Sale Of Investment":                               "投資有価証券の売却",
    "Net Investment Purchase And Sale":                 "投資の純取得・売却",
    "Purchase Of Business":                             "事業取得",
    "Sale Of Business":                                 "事業売却",
    "Net Business Purchase And Sale":                   "事業の純取得・売却",
    "Interest Received CFI":                            "受取利息（投資CF）",
    "Financing Cash Flow":                              "財務活動によるキャッシュフロー",
    "Net Issuance Payments Of Debt":                    "借入金の純増減",
    "Net Long Term Debt Issuance":                      "長期借入金の純増減",
    "Long Term Debt Issuance":                          "長期借入金の増加",
    "Long Term Debt Payments":                          "長期借入金の返済",
    "Net Short Term Debt Issuance":                     "短期借入金の純増減",
    "Issuance Of Debt":                                 "借入金の増加",
    "Repayment Of Debt":                                "借入金の返済",
    "Net Common Stock Issuance":                        "普通株式の純発行",
    "Common Stock Issuance":                            "普通株式の発行",
    "Common Stock Payments":                            "普通株式の取得",
    "Issuance Of Capital Stock":                        "株式の発行",
    "Repurchase Of Capital Stock":                      "自己株式取得",
    "Cash Dividends Paid":                              "配当金支払",
    "Common Stock Dividend Paid":                       "普通株配当金支払",
    "Proceeds From Stock Option Exercised":             "ストックオプション行使",
    "Free Cash Flow":                                   "フリーキャッシュフロー",
    "Beginning Cash Position":                          "期首現金残高",
    "End Cash Position":                                "期末現金残高",
    "Changes In Cash":                                  "現金の増減",
    "Effect Of Exchange Rate Changes":                  "為替変動の影響",
    "Income Tax Paid Supplemental Data":                "法人税支払（補足）",
    "Interest Paid Supplemental Data":                  "支払利息（補足）",
    "Taxes Refund Paid":                                "法人税の支払・還付",
    "Cash Flow From Continuing Financing Activities":   "継続事業財務活動によるキャッシュフロー",
    "Net Other Financing Charges":                      "その他財務活動純額",
    "Cash Flow From Continuing Investing Activities":   "継続事業投資活動によるキャッシュフロー",
    "Net Other Investing Changes":                      "その他投資活動純額",
    "Net Intangibles Purchase And Sale":                "無形固定資産の純取得・売却",
    "Purchase Of Intangibles":                          "無形固定資産の取得",
    "Sale Of PPE":                                      "有形固定資産の売却",
    "Purchase Of PPE":                                  "有形固定資産の取得",
    "Cash Flow From Continuing Operating Activities":   "継続事業営業活動によるキャッシュフロー",
    "Interest Received Cfo":                            "受取利息（営業CF）",
    "Interest Paid Cfo":                                "支払利息（営業CF）",
    "Dividend Received Cfo":                            "受取配当金（営業CF）",
    "Change In Other Working Capital":                  "その他運転資本の変動",
    "Change In Payables And Accrued Expense":           "買掛金・未払費用の変動",
    "Change In Payable":                                "支払債務の変動",
    "Deferred Income Tax":                              "繰延法人税",
    "Depreciation Amortization Depletion":              "減価償却・償却・枯渇費",
    "Operating Gains Losses":                           "営業損益",
    "Earnings Losses From Equity Investments":          "持分法投資損益",
    "Other Cash Adjustment Outside Changein Cash":      "現金変動外その他調整",
    "Pension And Employee Benefit Expense":             "退職給付・従業員給付費用",
    "Gain Loss On Investment Securities":               "投資有価証券損益",
    "Gain Loss On Sale Of PPE":                        "有形固定資産売却損益",
}

def translate_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = [FINANCIAL_TRANSLATIONS.get(str(i), str(i)) for i in df.index]
    return df

# 日足データ（スライスで対応）
DAILY_PERIODS = {
    "1ヶ月": 30,
    "3ヶ月": 90,
    "1年": 365,
    "5年": 365 * 5,
    "10年": 365 * 10,
}

# 分足・時間足データ（個別取得）
INTRADAY_PERIODS = {
    "1日":   {"period": "1d",  "interval": "5m",  "dt_fmt": "%H:%M"},
    "1週間": {"period": "5d",  "interval": "30m", "dt_fmt": "%m/%d"},
}

PERIODS = list(INTRADAY_PERIODS) + list(DAILY_PERIODS)

st.set_page_config(page_title="株価分析ダッシュボード", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 0.5rem; }
    h1 { font-size: 1.8rem !important; }
    h2, h3 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }
    [data-testid="metric-container"] { padding: 0.4rem 0.6rem !important; }
    [data-testid="stForm"] { border: none; padding: 0; }
    div[data-testid="stFormSubmitButton"] > button {
        height: 2.6rem;
        margin-top: 1.65rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("株価分析ダッシュボード")

tab_dashboard, tab_predict, tab_tomorrow, tab_financial, tab_search = st.tabs(["株価分析", "モデル検証", "明日の予測", "財務情報", "銘柄コード検索"])

# =====================================================================
# タブ1：株価分析
# =====================================================================
with tab_dashboard:

    with st.form("ticker_form"):
        input_col, btn_col = st.columns([6, 1])
        with input_col:
            ticker_input = st.text_input("銘柄コードを入力してください（例: AAPL, 7203.T）")
        with btn_col:
            submitted = st.form_submit_button("決定", use_container_width=True)

    if submitted and ticker_input:
        st.session_state["ticker"] = ticker_input.strip().upper()
        st.session_state["period"] = "3ヶ月"
        st.session_state["chart_type"] = "candle"
        st.session_state["hist_all"] = None
        st.session_state["info"] = None
        st.session_state["error"] = None

    loaded = "ticker" in st.session_state

    if loaded:
        ticker = st.session_state["ticker"]

        if st.session_state.get("hist_all") is None:
            with st.spinner(f"{ticker} のデータを取得中..."):
                stock = yf.Ticker(ticker, session=_session)
                info = stock.info
                hist_all = stock.history(period="10y")
                if hist_all.empty:
                    st.session_state["error"] = "データを取得できませんでした。銘柄コードを確認してください。"
                else:
                    st.session_state["info"] = info
                    st.session_state["hist_all"] = hist_all
                    st.session_state["error"] = None

        if st.session_state.get("error"):
            st.error(st.session_state["error"])
            loaded = False

    # 銘柄名
    if loaded:
        info = st.session_state["info"]
        ticker = st.session_state["ticker"]
        st.subheader(f"{info.get('longName', ticker)} ({ticker})")
    else:
        st.subheader("ーー")

    # 指標カード
    col1, col2, col3, col4 = st.columns(4)

    if loaded:
        info = st.session_state["info"]
        market_cap = info.get("marketCap")
        per = info.get("trailingPE")
        pbr = info.get("priceToBook")
        dividend_yield = info.get("dividendYield")
        currency = info.get("currency", "JPY")

        # 通貨に応じた単位ラベルと区切り値
        if currency == "JPY":
            unit_large, unit_small = "兆円", "億円"
            thresh_large, thresh_small = 1_000_000_000_000, 100_000_000
            div_large,    div_small   = 1_000_000_000_000, 100_000_000
        else:
            unit_large, unit_small = "兆ドル", "億ドル"
            thresh_large, thresh_small = 1_000_000_000_000, 100_000_000
            div_large,    div_small   = 1_000_000_000_000, 100_000_000

        with col1:
            if market_cap:
                if market_cap >= thresh_large:
                    cap_str = f"{market_cap / div_large:.2f} {unit_large}"
                elif market_cap >= thresh_small:
                    cap_str = f"{market_cap / div_small:.0f} {unit_small}"
                else:
                    cap_str = f"{market_cap:,}"
                st.metric("時価総額", cap_str)
            else:
                st.metric("時価総額", "N/A")
        with col2:
            st.metric("PER（株価収益率）", f"{per:.2f} 倍" if per else "N/A")
        with col3:
            st.metric("PBR（株価純資産倍率）", f"{pbr:.2f} 倍" if pbr else "N/A")
        with col4:
            if dividend_yield:
                # yfinanceは全通貨でパーセント値を返す（0.36 → 0.36%、3.38 → 3.38%）
                st.metric("配当利回り", f"{dividend_yield:.2f} %")
            else:
                st.metric("配当利回り", "N/A")
    else:
        with col1:
            st.metric("時価総額", "ーー")
        with col2:
            st.metric("PER（株価収益率）", "ーー")
        with col3:
            st.metric("PBR（株価純資産倍率）", "ーー")
        with col4:
            st.metric("配当利回り", "ーー")

    # 株価推移セクション
    st.subheader("株価推移")

    if loaded:
        ticker = st.session_state["ticker"]
        selected_period = st.session_state.get("period", "3ヶ月")

        # データ準備
        if selected_period in INTRADAY_PERIODS:
            cfg = INTRADAY_PERIODS[selected_period]
            cache_key = f"{ticker}_{selected_period}"
            if st.session_state.get("intraday_key") != cache_key:
                with st.spinner(f"{selected_period}のデータを取得中..."):
                    stk = yf.Ticker(ticker, session=_session)
                    intraday = stk.history(period=cfg["period"], interval=cfg["interval"])
                    st.session_state["intraday_hist"] = intraday
                    st.session_state["intraday_key"] = cache_key
            hist = st.session_state["intraday_hist"].copy()
            dt_fmt = cfg["dt_fmt"]
        else:
            hist_all = st.session_state["hist_all"]
            cutoff = datetime.today() - timedelta(days=DAILY_PERIODS[selected_period])
            hist = hist_all[hist_all.index >= cutoff.strftime("%Y-%m-%d")].copy()
            dt_fmt = "%m/%d"

        candle_col, bb_col, period_col = st.columns([3.5, 3.5, 0.7])

        # 期間ボタン
        with period_col:
            st.write("")
            for label in PERIODS:
                st.button(
                    label,
                    use_container_width=True,
                    type="primary" if st.session_state.get("period") == label else "secondary",
                    on_click=lambda l=label: st.session_state.update({"period": l}),
                )

        # ── 左：ローソク足 + MA ──────────────────────
        with candle_col:
            import matplotlib.lines as mlines
            MA5_COLOR  = "#00bfff"
            MA25_COLOR = "#bf7fff"

            df_mpf = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            df_mpf.index = pd.DatetimeIndex(df_mpf.index).tz_localize(None)

            mc = mpf.make_marketcolors(
                up="red", down="green",
                edge="inherit", wick="inherit", volume="inherit",
            )
            dark_style = mpf.make_mpf_style(
                base_mpf_style="nightclouds",
                marketcolors=mc,
                mavcolors=[MA5_COLOR, MA25_COLOR],
                facecolor="#0e1117",
                figcolor="#0e1117",
                edgecolor="#0e1117",
                gridcolor="#2d2d2d",
                gridstyle="-",
                rc={"axes.labelcolor": "#ffffff", "xtick.color": "#aaaaaa", "ytick.color": "#aaaaaa"},
            )
            fig, axes = mpf.plot(
                df_mpf,
                type="candle",
                mav=(5, 25),
                volume=False,
                style=dark_style,
                datetime_format=dt_fmt,
                figsize=(6, 3.5),
                returnfig=True,
            )
            ma5_h  = mlines.Line2D([], [], color=MA5_COLOR,  linewidth=1.5, label="MA5")
            ma25_h = mlines.Line2D([], [], color=MA25_COLOR, linewidth=1.5, label="MA25")
            axes[0].legend(handles=[ma5_h, ma25_h], loc="upper left",
                           facecolor="#1c1c2e", labelcolor="white",
                           framealpha=0.7, fontsize=9)
            st.pyplot(fig)
            plt.close(fig)

        # ── 右：ボリンジャーバンド + 終値 ────────────
        with bb_col:
            if selected_period in INTRADAY_PERIODS:
                bb_src = hist["Close"]
            else:
                bb_src = st.session_state["hist_all"]["Close"]

            ma20  = bb_src.rolling(20).mean()
            std20 = bb_src.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20

            cutoff_idx = pd.DatetimeIndex(hist.index).tz_localize(None)
            close = pd.Series(hist["Close"].values, index=cutoff_idx)
            ma20_v  = ma20.reindex(hist.index).values
            upper_v = upper.reindex(hist.index).values
            lower_v = lower.reindex(hist.index).values

            import matplotlib.dates as mdates
            fig_bb, ax_bb = plt.subplots(figsize=(6, 3.5))
            fig_bb.patch.set_facecolor("#0e1117")
            ax_bb.set_facecolor("#0e1117")
            ax_bb.tick_params(colors="#aaaaaa")
            for spine in ax_bb.spines.values():
                spine.set_edgecolor("#2d2d2d")
            ax_bb.grid(True, color="#2d2d2d", linewidth=0.5)
            ax_bb.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax_bb.xaxis.set_major_locator(mdates.AutoDateLocator())

            ax_bb.fill_between(cutoff_idx, upper_v, lower_v, alpha=0.12, color="#bf7fff")
            ax_bb.plot(cutoff_idx, upper_v, color="#bf7fff", linewidth=0.8,
                       linestyle="--", label="Upper Band")
            ax_bb.plot(cutoff_idx, lower_v, color="#bf7fff", linewidth=0.8,
                       linestyle="--", label="Lower Band")
            ax_bb.plot(cutoff_idx, ma20_v,  color="#ffb74d", linewidth=1.0,
                       linestyle="--", label="MA20")
            ax_bb.plot(cutoff_idx, close.values, color="#4fc3f7", linewidth=1.5,
                       label="Close")
            ax_bb.legend(facecolor="#1c1c2e", labelcolor="white",
                         framealpha=0.7, fontsize=9, loc="upper left")
            plt.tight_layout()
            st.pyplot(fig_bb)
            plt.close(fig_bb)

    else:
        chart_col, _ = st.columns([6, 1])
        with chart_col:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            fig.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#0e1117")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d2d2d")
            ax.set_xticks([])
            ax.set_yticks([])
            st.pyplot(fig)
            plt.close(fig)
        plt.close(fig)

# =====================================================================
# =====================================================================
# タブ2：モデル検証
# =====================================================================
with tab_predict:

    if not st.session_state.get("info"):
        st.info("ダッシュボードタブで銘柄コードを入力・決定してください。")
    else:
        info = st.session_state["info"]
        ticker = st.session_state["ticker"]
        st.subheader(f"{info.get('longName', ticker)} ({ticker})")

        ticker_sym = st.session_state["ticker"]

        # 銘柄が変わったときだけ再学習
        if st.session_state.get("predict_ticker") != ticker_sym:
            with st.spinner("予測モデルを学習中...（数秒かかります）"):
                df_pred = yf.download(ticker_sym, period="5y", interval="1d", auto_adjust=True, progress=False, session=_session)
                if isinstance(df_pred.columns, pd.MultiIndex):
                    df_pred.columns = df_pred.columns.droplevel(1)

                # ── 特徴量エンジニアリング ──────────────────
                close  = df_pred["Close"]
                high   = df_pred["High"]
                low    = df_pred["Low"]
                volume = df_pred["Volume"]

                # ラグ特徴量（過去5日の終値）
                for i in range(1, 6):
                    df_pred[f"Lag_{i}"] = close.shift(i)

                # 移動平均比率
                ma5  = close.rolling(5).mean()
                ma25 = close.rolling(25).mean()
                df_pred["MA5_ratio"]  = close / ma5
                df_pred["MA25_ratio"] = close / ma25

                # RSI（14日）
                delta     = close.diff()
                gain      = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss      = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                df_pred["RSI14"] = 100 - (100 / (1 + gain / loss))

                # MACD
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd  = ema12 - ema26
                macd_signal = macd.ewm(span=9, adjust=False).mean()
                df_pred["MACD"]        = macd
                df_pred["MACD_signal"] = macd_signal
                df_pred["MACD_hist"]   = macd - macd_signal

                # ボリンジャーバンド位置（0〜1）
                ma20  = close.rolling(20).mean()
                std20 = close.rolling(20).std()
                upper = ma20 + 2 * std20
                lower = ma20 - 2 * std20
                df_pred["BB_position"] = (close - lower) / (upper - lower)

                # 出来高比率
                vol_ma5 = volume.rolling(5).mean()
                df_pred["Volume_ratio"] = volume / vol_ma5

                # ボラティリティ（高値−安値比率）
                df_pred["HL_ratio"] = (high - low) / close

                # 短期リターン
                df_pred["Return_1"] = close.pct_change(1)
                df_pred["Return_5"] = close.pct_change(5)

                # カレンダー特徴量
                df_pred["DayOfWeek"] = df_pred.index.dayofweek  # 0=月曜 〜 4=金曜
                df_pred["Month"]     = df_pred.index.month      # 1〜12

                df_pred = df_pred.dropna()

                feature_cols = (
                    [f"Lag_{i}" for i in range(1, 6)]
                    + ["MA5_ratio", "MA25_ratio", "RSI14",
                       "MACD", "MACD_signal", "MACD_hist",
                       "BB_position", "Volume_ratio", "HL_ratio",
                       "Return_1", "Return_5",
                       "DayOfWeek", "Month"]
                )
                X = df_pred[feature_cols]
                y = df_pred["Close"]

                split = int(len(df_pred) * 0.8)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]
                test_dates = df_pred.index[split:]

                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                y_arr   = y_test.values.ravel()
                pred_arr = predictions.flatten()

                rmse = np.sqrt(mean_squared_error(y_arr, pred_arr))
                mae  = mean_absolute_error(y_arr, pred_arr)
                r2   = r2_score(y_arr, pred_arr)

                # 投資シミュレーション
                sim = pd.DataFrame({"Actual": y_arr, "Predicted": pred_arr}, index=test_dates)
                sim["Predicted_Change"] = sim["Predicted"] - sim["Actual"].shift(1)
                sim["Signal"]           = np.where(sim["Predicted_Change"] > 0, 1, 0)
                sim["Actual_Returns"]   = sim["Actual"].pct_change()
                sim["Strategy_Returns"] = sim["Signal"] * sim["Actual_Returns"]

                cum_strategy = (1 + sim["Strategy_Returns"]).cumprod()
                cum_buyhold  = (1 + sim["Actual_Returns"]).cumprod()

                # インデックス比較（日本株→日経225、それ以外→S&P500）
                index_sym  = "^N225" if ticker_sym.endswith(".T") else "^GSPC"
                index_name = "Nikkei 225" if ticker_sym.endswith(".T") else "S&P 500"
                idx_raw = yf.download(index_sym, period="5y", interval="1d",
                                      auto_adjust=True, progress=False, session=_session)
                if isinstance(idx_raw.columns, pd.MultiIndex):
                    idx_raw.columns = idx_raw.columns.droplevel(1)
                idx_close = idx_raw["Close"].reindex(test_dates, method="ffill").dropna()
                idx_returns = idx_close.pct_change().fillna(0)
                cum_index = (1 + idx_returns).cumprod()
                # テスト期間の日付に揃える
                cum_index = cum_index.reindex(cum_buyhold.index, method="ffill")

                st.session_state.update({
                    "predict_ticker":       ticker_sym,
                    "predict_rmse":         rmse,
                    "predict_mae":          mae,
                    "predict_r2":           r2,
                    "predict_cum_strategy": cum_strategy,
                    "predict_cum_buyhold":  cum_buyhold,
                    "predict_cum_index":    cum_index,
                    "predict_index_name":   index_name,
                    "predict_test_dates":   test_dates,
                    "predict_y_arr":        y_arr,
                    "predict_pred_arr":     pred_arr,
                    "predict_sim":          sim,
                    "predict_model":        model,
                    "predict_feature_cols": feature_cols,
                })

        # ── 評価指標 ──────────────────────────────────
        rmse = st.session_state["predict_rmse"]
        mae  = st.session_state["predict_mae"]
        r2   = st.session_state["predict_r2"]

        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE（誤差の大きさ）", f"{rmse:.2f}")
        m2.metric("MAE（平均絶対誤差）",   f"{mae:.2f} 円")
        m3.metric("R²（決定係数）",        f"{r2:.4f}")

        # ── 上半分：学習データの合致度合い ───────────
        st.subheader("予測と実績の比較")

        test_dates = st.session_state["predict_test_dates"]
        y_arr      = st.session_state["predict_y_arr"]
        pred_arr   = st.session_state["predict_pred_arr"]

        actual_diff    = y_arr[1:] - y_arr[:-1]
        predicted_diff = pred_arr[1:] - y_arr[:-1]

        fig2, (ax_ts, ax_sc) = plt.subplots(1, 2, figsize=(12, 3.5))
        for ax in (ax_ts, ax_sc):
            ax.set_facecolor("#0e1117")
            ax.tick_params(colors="#aaaaaa")
            for spine in ax.spines.values():
                spine.set_edgecolor("#2d2d2d")
            ax.grid(True, color="#2d2d2d", linewidth=0.5)
        fig2.patch.set_facecolor("#0e1117")

        # 時系列グラフ
        ax_ts.plot(test_dates, y_arr,    label="Actual",      color="#4fc3f7", alpha=0.8, linewidth=1.5)
        ax_ts.plot(test_dates, pred_arr, label="AI Prediction", color="#ef5350",
                   linestyle="--", alpha=0.9, linewidth=1.5)
        ax_ts.set_title(f"Actual vs AI Prediction  (MAE: {mae:.2f})", color="white")
        ax_ts.set_xlabel("Date", color="#aaaaaa")
        ax_ts.set_ylabel("Price", color="#aaaaaa")
        ax_ts.legend(facecolor="#1c1c2e", labelcolor="white", framealpha=0.7)

        # 散布図
        ax_sc.scatter(actual_diff, predicted_diff, alpha=0.4, color="#ce93d8", s=15)
        ax_sc.axhline(0, color="#aaaaaa", linewidth=0.8)
        ax_sc.axvline(0, color="#aaaaaa", linewidth=0.8)
        min_v = min(actual_diff.min(), predicted_diff.min())
        max_v = max(actual_diff.max(), predicted_diff.max())
        ax_sc.plot([min_v, max_v], [min_v, max_v], color="#ef5350",
                   linestyle="--", linewidth=1.5, label="Perfect Fit")
        ax_sc.set_title("Predicted vs Actual Daily Change", color="white")
        ax_sc.set_xlabel("Actual Change", color="#aaaaaa")
        ax_sc.set_ylabel("Predicted Change", color="#aaaaaa")
        ax_sc.legend(facecolor="#1c1c2e", labelcolor="white", framealpha=0.7)

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

        # ── 下半分：投資シミュレーション ──────────────
        st.subheader("投資シミュレーション（バックテスト）")

        cum_strategy = st.session_state["predict_cum_strategy"]
        cum_buyhold  = st.session_state["predict_cum_buyhold"]
        cum_index    = st.session_state["predict_cum_index"]
        index_name   = st.session_state["predict_index_name"]
        sim          = st.session_state["predict_sim"]

        final_strategy = (cum_strategy.iloc[-1] - 1) * 100
        final_buyhold  = (cum_buyhold.iloc[-1]  - 1) * 100
        final_index    = (cum_index.iloc[-1]    - 1) * 100

        graph_col, table_col = st.columns([3, 2])

        with graph_col:
            fig_sim, ax_sim = plt.subplots(figsize=(9, 4))
            fig_sim.patch.set_facecolor("#0e1117")
            ax_sim.set_facecolor("#0e1117")
            ax_sim.plot(cum_strategy.index, cum_strategy.values,
                        label=f"AI Strategy  {final_strategy:+.1f}%", color="#00e676", linewidth=2)
            ax_sim.plot(cum_buyhold.index, cum_buyhold.values,
                        label=f"Buy & Hold  {final_buyhold:+.1f}%", color="#aaaaaa",
                        linestyle=":", linewidth=2)
            ax_sim.plot(cum_index.index, cum_index.values,
                        label=f"{index_name}  {final_index:+.1f}%", color="#ffb74d",
                        linestyle="--", linewidth=1.5)
            ax_sim.set_xlabel("Date", color="#aaaaaa")
            ax_sim.set_ylabel("Cumulative Return", color="#aaaaaa")
            ax_sim.tick_params(colors="#aaaaaa")
            for spine in ax_sim.spines.values():
                spine.set_edgecolor("#2d2d2d")
            ax_sim.legend(facecolor="#1c1c2e", labelcolor="white", framealpha=0.7)
            ax_sim.grid(True, color="#2d2d2d", linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig_sim)
            plt.close(fig_sim)

        with table_col:
            st.caption("売買履歴（買い / 待機）")
            trade_log = sim.copy()
            trade_log.index = trade_log.index.strftime("%Y-%m-%d")
            trade_log["売買"]          = np.where(trade_log["Signal"] == 1, "買い", "待機")
            trade_log["価格"]          = trade_log["Actual"].round(0).astype(int)
            trade_log["当日リターン(%)"] = (trade_log["Actual_Returns"] * 100).round(2)
            trade_log["戦略リターン(%)"] = (trade_log["Strategy_Returns"] * 100).round(2)
            st.dataframe(
                trade_log[["売買", "価格", "当日リターン(%)", "戦略リターン(%)"]],
                use_container_width=True,
                height=300,
            )

# =====================================================================
# タブ3：明日の予測
# =====================================================================
with tab_tomorrow:
    st.subheader("明日の予測シグナル")

    if not st.session_state.get("predict_model"):
        st.info("先に「モデル検証」タブを開いてモデルを学習させてください。")
    else:
        model_rf      = st.session_state["predict_model"]
        feature_cols  = st.session_state["predict_feature_cols"]
        ticker_sym    = st.session_state["ticker"]

        with st.spinner("最新データを取得して予測中..."):
            df_lt = yf.download(ticker_sym, period="3mo", interval="1d",
                                auto_adjust=True, progress=False, session=_session)
            if isinstance(df_lt.columns, pd.MultiIndex):
                df_lt.columns = df_lt.columns.droplevel(1)

            c  = df_lt["Close"]
            h  = df_lt["High"]
            lo = df_lt["Low"]
            v  = df_lt["Volume"]

            for i in range(1, 6):
                df_lt[f"Lag_{i}"] = c.shift(i)

            ma5  = c.rolling(5).mean()
            ma25 = c.rolling(25).mean()
            df_lt["MA5_ratio"]  = c / ma5
            df_lt["MA25_ratio"] = c / ma25

            delta = c.diff()
            gain  = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            df_lt["RSI14"] = 100 - (100 / (1 + gain / loss))

            ema12 = c.ewm(span=12, adjust=False).mean()
            ema26 = c.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            macd_sig = macd.ewm(span=9, adjust=False).mean()
            df_lt["MACD"]        = macd
            df_lt["MACD_signal"] = macd_sig
            df_lt["MACD_hist"]   = macd - macd_sig

            ma20  = c.rolling(20).mean()
            std20 = c.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            df_lt["BB_position"]  = (c - lower) / (upper - lower)
            df_lt["Volume_ratio"] = v / v.rolling(5).mean()
            df_lt["HL_ratio"]     = (h - lo) / c
            df_lt["Return_1"]     = c.pct_change(1)
            df_lt["Return_5"]     = c.pct_change(5)
            df_lt["DayOfWeek"]    = df_lt.index.dayofweek
            df_lt["Month"]        = df_lt.index.month

            df_lt = df_lt.dropna()

            latest_features   = df_lt[feature_cols].iloc[[-1]]
            predicted_price   = model_rf.predict(latest_features)[0]
            today_price       = float(df_lt["Close"].iloc[-1])
            predicted_change  = (predicted_price - today_price) / today_price * 100
            signal            = "買い" if predicted_price > today_price else "待機"
            today_date        = df_lt.index[-1].strftime("%Y-%m-%d")

        st.caption(f"基準日: {today_date}　（翌営業日の終値を予測）")

        sig_col, price_col, pred_col = st.columns(3)
        with sig_col:
            signal_label = "買い" if signal == "買い" else "待機"
            st.metric("シグナル", signal_label)
        with price_col:
            st.metric("現在の株価", f"{today_price:,.0f}")
        with pred_col:
            st.metric("予測株価（翌日終値）", f"{predicted_price:,.0f}",
                      delta=f"{predicted_change:+.2f}%")

        # 直近30日の株価チャート + 予測点
        st.subheader("直近の株価と予測")
        df_chart = df_lt["Close"].tail(30)

        fig_tm, ax_tm = plt.subplots(figsize=(12, 3.5))
        fig_tm.patch.set_facecolor("#0e1117")
        ax_tm.set_facecolor("#0e1117")
        ax_tm.tick_params(colors="#aaaaaa")
        for spine in ax_tm.spines.values():
            spine.set_edgecolor("#2d2d2d")
        ax_tm.grid(True, color="#2d2d2d", linewidth=0.5)

        ax_tm.plot(df_chart.index, df_chart.values,
                   color="#4fc3f7", linewidth=1.5, label="Actual")

        pred_color = "#ef5350" if signal == "待機" else "#00e676"
        next_date  = df_chart.index[-1] + pd.tseries.offsets.BDay(1)
        ax_tm.scatter([next_date], [predicted_price],
                      color=pred_color, s=80, zorder=5,
                      label=f"Prediction  {predicted_change:+.2f}%")
        ax_tm.plot([df_chart.index[-1], next_date],
                   [today_price, predicted_price],
                   color=pred_color, linestyle="--", linewidth=1.2)

        ax_tm.set_xlabel("Date", color="#aaaaaa")
        ax_tm.set_ylabel("Price", color="#aaaaaa")
        ax_tm.legend(facecolor="#1c1c2e", labelcolor="white", framealpha=0.7)
        plt.tight_layout()
        st.pyplot(fig_tm)
        plt.close(fig_tm)

# =====================================================================
# タブ3：財務情報
# =====================================================================
with tab_financial:

    if not st.session_state.get("info"):
        st.info("ダッシュボードタブで銘柄コードを入力・決定してください。")
    else:
        fin_ticker = st.session_state["ticker"]
        fin_name   = st.session_state["info"].get("longName", fin_ticker)
        st.subheader(f"{fin_name}（{fin_ticker}）の財務情報")

        # 取得済みでなければフェッチ
        if st.session_state.get("fin_ticker_loaded") != fin_ticker:
            with st.spinner("財務データを取得中..."):
                stk = yf.Ticker(fin_ticker, session=_session)
                st.session_state["income_stmt"]   = stk.income_stmt
                st.session_state["balance_sheet"]  = stk.balance_sheet
                st.session_state["cashflow"]       = stk.cashflow
                st.session_state["news_data"]      = stk.news
                st.session_state["fin_ticker_loaded"] = fin_ticker

        def fmt_df(df: pd.DataFrame) -> pd.DataFrame:
            """列名を年表示にし、項目名を日本語化し、数値を億円単位に変換して返す"""
            if df is None or df.empty:
                return pd.DataFrame()
            df = translate_index(df)
            df.columns = [c.strftime("%Y年") if hasattr(c, "strftime") else str(c) for c in df.columns]
            df.index.name = "項目"
            numeric_cols = df.select_dtypes(include="number").columns
            df[numeric_cols] = (df[numeric_cols] / 1_0000_0000).round(1)
            return df

        sub_is, sub_bs, sub_cf, sub_news = st.tabs(
            ["損益計算書", "バランスシート", "キャッシュフロー", "ニュース"]
        )

        with sub_is:
            df_is = fmt_df(st.session_state["income_stmt"])
            if df_is.empty:
                st.info("損益計算書のデータがありません。")
            else:
                st.caption("単位：億円")
                st.dataframe(df_is, use_container_width=True)

        with sub_bs:
            df_bs = fmt_df(st.session_state["balance_sheet"])
            if df_bs.empty:
                st.info("バランスシートのデータがありません。")
            else:
                st.caption("単位：億円")
                st.dataframe(df_bs, use_container_width=True)

        with sub_cf:
            df_cf = fmt_df(st.session_state["cashflow"])
            if df_cf.empty:
                st.info("キャッシュフローのデータがありません。")
            else:
                st.caption("単位：億円")
                st.dataframe(df_cf, use_container_width=True)

        with sub_news:
            news_list = st.session_state.get("news_data") or []
            if not news_list:
                st.info("ニュースが見つかりませんでした。")
            else:
                for item in news_list:
                    # yfinance 1.x では content キーにネストされている場合がある
                    content = item.get("content", item)
                    title      = content.get("title", "（タイトルなし）")
                    publisher  = content.get("provider", {}).get("displayName", "") if isinstance(content.get("provider"), dict) else content.get("publisher", "")
                    url        = content.get("canonicalUrl", {}).get("url", "") if isinstance(content.get("canonicalUrl"), dict) else content.get("link", "")
                    pub_time   = content.get("pubDate") or content.get("providerPublishTime")
                    if isinstance(pub_time, (int, float)):
                        pub_time = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d %H:%M")

                    with st.container(border=True):
                        if url:
                            st.markdown(f"**[{title}]({url})**")
                        else:
                            st.markdown(f"**{title}**")
                        meta = " ｜ ".join(filter(None, [publisher, str(pub_time) if pub_time else ""]))
                        if meta:
                            st.caption(meta)

# =====================================================================
# タブ3：銘柄コード検索
# =====================================================================
with tab_search:

    st.subheader("銘柄コード検索")
    st.caption("企業名・キーワードで検索すると銘柄コードを確認できます（例：Toyota、Apple、Sony）")

    with st.form("search_form"):
        s_col, b_col = st.columns([6, 1])
        with s_col:
            search_query = st.text_input("企業名・キーワードを入力してください")
        with b_col:
            search_submitted = st.form_submit_button("検索", use_container_width=True)

    if search_submitted and search_query:
        with st.spinner("検索中..."):
            try:
                results = yf.Search(search_query, max_results=20, session=_session)
                quotes = results.quotes
            except Exception as e:
                quotes = []
                st.error(f"検索中にエラーが発生しました: {e}")

        if quotes:
            rows = []
            for q in quotes:
                rows.append({
                    "銘柄コード": q.get("symbol", ""),
                    "企業名": q.get("longname") or q.get("shortname", ""),
                    "取引所": q.get("exchange", ""),
                    "種別": q.get("quoteType", ""),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(df)} 件見つかりました。銘柄コードをダッシュボードタブに入力してご利用ください。")
        else:
            st.info("該当する銘柄が見つかりませんでした。別のキーワードで試してください。")
