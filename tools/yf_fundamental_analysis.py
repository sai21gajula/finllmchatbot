import yfinance as yf
import pandas as pd
from datetime import datetime
from crewai.tools import BaseTool

class YFinanceFundamentalAnalysisTool(BaseTool):
    """
    A BaseTool implementation for performing fundamental analysis on a given stock symbol.
    """

    def __init__(self):
        super().__init__(
            name="yf_fundamental_analysis",
            description="Perform a comprehensive fundamental analysis on the given stock symbol."
        )

    def _run(self, ticker: str) -> dict:
        """
        Perform the fundamental analysis.

        Args:
            ticker (str): The stock symbol to analyze.

        Returns:
            dict: A dictionary with the detailed fundamental analysis results.
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Financial Data
            financials = stock.financials.infer_objects(copy=False)
            balance_sheet = stock.balance_sheet.infer_objects(copy=False)
            cash_flow = stock.cashflow.infer_objects(copy=False)

            # Fill missing values
            financials = financials.ffill()
            balance_sheet = balance_sheet.ffill()
            cash_flow = cash_flow.ffill()

            # Key Ratios and Metrics
            ratios = {
                "P/E Ratio": info.get('trailingPE'),
                "Forward P/E": info.get('forwardPE'),
                "P/B Ratio": info.get('priceToBook'),
                "P/S Ratio": info.get('priceToSalesTrailing12Months'),
                "PEG Ratio": info.get('pegRatio'),
                "Debt to Equity": info.get('debtToEquity'),
                "Current Ratio": info.get('currentRatio'),
                "Quick Ratio": info.get('quickRatio'),
                "ROE": info.get('returnOnEquity'),
                "ROA": info.get('returnOnAssets'),
                "ROIC": info.get('returnOnCapital'),
                "Gross Margin": info.get('grossMargins'),
                "Operating Margin": info.get('operatingMargins'),
                "Net Profit Margin": info.get('profitMargins'),
                "Dividend Yield": info.get('dividendYield'),
                "Payout Ratio": info.get('payoutRatio'),
            }

            # Growth Rates
            revenue = financials.loc['Total Revenue']
            net_income = financials.loc['Net Income']
            revenue_growth = revenue.pct_change(periods=-1).iloc[0] if len(revenue) > 1 else None
            net_income_growth = net_income.pct_change(periods=-1).iloc[0] if len(net_income) > 1 else None

            growth_rates = {
                "Revenue Growth (YoY)": revenue_growth,
                "Net Income Growth (YoY)": net_income_growth,
            }

            # Valuation
            market_cap = info.get('marketCap')
            enterprise_value = info.get('enterpriseValue')

            valuation = {
                "Market Cap": market_cap,
                "Enterprise Value": enterprise_value,
                "EV/EBITDA": info.get('enterpriseToEbitda'),
                "EV/Revenue": info.get('enterpriseToRevenue'),
            }

            # Future Estimates
            estimates = {
                "Next Year EPS Estimate": info.get('forwardEps'),
                "Next Year Revenue Estimate": info.get('revenueEstimates', {}).get('avg'),
                "Long-term Growth Rate": info.get('longTermPotentialGrowthRate'),
            }

            # Simple DCF Valuation
            free_cash_flow = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else None
            wacc = 0.1  # Assumed Weighted Average Cost of Capital
            growth_rate = info.get('longTermPotentialGrowthRate', 0.03)

            def simple_dcf(fcf, growth_rate, wacc, years=5):
                if fcf is None or growth_rate is None:
                    return None
                terminal_value = fcf * (1 + growth_rate) / (wacc - growth_rate)
                dcf_value = sum([fcf * (1 + growth_rate) ** i / (1 + wacc) ** i for i in range(1, years + 1)])
                dcf_value += terminal_value / (1 + wacc) ** years
                return dcf_value

            dcf_value = simple_dcf(free_cash_flow, growth_rate, wacc)

            # Prepare Results
            analysis = {
                "Company Name": info.get('longName'),
                "Sector": info.get('sector'),
                "Industry": info.get('industry'),
                "Key Ratios": ratios,
                "Growth Rates": growth_rates,
                "Valuation Metrics": valuation,
                "Future Estimates": estimates,
                "Simple DCF Valuation": dcf_value,
                "Last Updated": datetime.fromtimestamp(info.get('lastFiscalYearEnd', 0)).strftime('%Y-%m-%d'),
                "Data Retrieval Date": datetime.now().strftime('%Y-%m-%d'),
            }

            return analysis

        except Exception as e:
            return {"error": f"An error occurred during the analysis: {str(e)}"}
# if __name__ == "__main__":
#     tool = YFinanceFundamentalAnalysisTool()
#     result = tool.run(ticker="AAPL")
#     print(result)