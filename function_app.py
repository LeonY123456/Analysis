import logging
import azure.functions as func
from stock_analysis import StockAnalysis  

app = func.FunctionApp()

@app.schedule(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def cook(myTimer: func.TimerRequest) -> None:
    
    if myTimer.past_due:
        logging.info('The timer is past due!')
    
    logging.info('Python timer trigger function executed.')

@app.route(route="graph", auth_level=func.AuthLevel.ANONYMOUS)
def graph(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # Get the stock name from the query parameters or request body
    stock_name = req.params.get('stock')
    if not stock_name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            stock_name = req_body.get('stock')

    if stock_name:
        try:
            # Create an instance of StockAnalysis for the given stock
            analysis = StockAnalysis(stock=stock_name)

            # Run the analysis steps
            df = analysis.get_stock_data()
            df = analysis.ema_strategy()       # Using the EMA strategy for example
            df = analysis.buy_sell_signals()   # Get buy/sell signals
            #df = analysis.backtest()           # Perform backtesting
            analysis.vcp_strategy()            # Run the VCP strategy

            # Render the graph as HTML and return it
            html_plot = analysis.render_plotly_plot()
            return func.HttpResponse(html_plot, mimetype="text/html", status_code=200)

        except Exception as e:
            logging.error(f"Error processing stock analysis for {stock_name}: {e}")
            return func.HttpResponse(f"Error processing analysis for {stock_name}.", status_code=500)
    
    else:
        return func.HttpResponse(
             "Please provide a stock name in the query string or request body.",
             status_code=400
        )