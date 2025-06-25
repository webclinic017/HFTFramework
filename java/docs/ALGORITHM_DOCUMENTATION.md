# Algorithm.java - Core Trading Algorithm Documentation

## Overview

The `Algorithm` class is the abstract base class for all trading algorithms in the Lambda Investing framework. It serves
as the central component that receives market data, sends orders, processes execution reports, and manages positions and
P&L. This class implements a comprehensive event-driven architecture for algorithmic trading.

## Key Interfaces

- `MarketDataListener`: Processes market data events (depth, trades, commands)
- `ExecutionReportListener`: Handles order execution updates
- `CandleListener`: Processes candle/bar data updates

## Algorithm Lifecycle

### Initialization

1. `constructorForAbstract()`: Base initialization of internal structures
2. `init()`: Registers with market data and trading engine connectors
3. `setParameters()`: Configures algorithm parameters

### Start/Stop

1. `start()`: Activates the algorithm to begin processing events and trading
2. `stop()`: Deactivates the algorithm, cancels all orders
3. `resetAlgorithm()`: Resets internal state for a fresh start

### State Management

- `algorithmState`: Tracks algorithm state (NOT_INITIALIZED, INITIALIZING, INITIALIZED, STARTING, STARTED, STOPPING,
  STOPPED)
- `checkOperationalTime()`: Manages trading hours based on firstHourOperatingIncluded and lastHourOperatingIncluded

## Market Data Processing

### Event Handlers

- `onDepthUpdate(Depth)`: Processes order book updates
- `onTradeUpdate(Trade)`: Processes market trade updates
- `onCommandUpdate(Command)`: Handles system commands (start, stop)
- `onCandleUpdate(Candle)`: Processes time-based candle/bar data

### Key Features

- Maintains last depth and trade for each instrument
- Updates internal time service based on message timestamps
- Filters out stale or duplicate updates
- Notifies registered observers of market data events

## Order Management

### Creating Orders

- `createLimitOrderRequest()`: Creates limit orders
- `createMarketOrderRequest()`: Creates market orders
- `createCancel()`: Creates cancel requests
- `generateClientOrderId()`: Generates unique order IDs

### Sending Orders

- `sendOrderRequest(OrderRequest)`: Validates and sends orders
- `checkOrderRequest(OrderRequest)`: Validates order parameters
- `sendQuoteRequest(QuoteRequest)`: Sends quotes for market making

### Managing Active Orders

- `updateAllActiveOrders(ExecutionReport)`: Updates internal order state
- `cancelAll(Instrument)`: Cancels all active orders for an instrument
- `cancelAllVerb(Instrument, Verb)`: Cancels all buy or sell orders
- `clientOrderIdToCancelWhenActive`: Queue for canceling orders when they become active

## Position Management

### Tracking Positions

- `addPosition(ExecutionReport)`: Updates position on fills
- `getPosition(Instrument)`: Gets current position for an instrument
- `getAlgorithmPosition(Instrument)`: Gets algorithm-specific position
- `onPosition(Map<String, Double>)`: Updates positions from external source
- `requestUpdatePosition(boolean)`: Requests position update from broker

## Portfolio and P&L Management

### Portfolio

- `portfolioManager`: Manages portfolio and P&L calculations
- `addToPersist(ExecutionReport)`: Adds trades to P&L calculation
- `getLastPnlSnapshot(String)`: Gets latest P&L snapshot
- `printSummaryResults()`: Outputs trading results

### Hedging

- `hedgeManager`: Manages hedging operations
- `setHedgeManager(HedgeManager)`: Sets custom hedge manager

## Backtest-Specific Features

### Backtest Management

- `isBacktest`: Flag for backtest mode
- `saveBacktestOutputTrades`: Controls saving trade outputs
- `printSummaryBacktest`: Controls printing summary
- `onFinishedBacktest()`: Handles backtest completion
- `plotBacktestResults()`: Plots backtest results
- `saveBacktestTrades()`: Saves backtest trades to file

### Time Management

- `timeService`: Provides current time (real or simulated)
- `getCurrentTime()`, `getCurrentTimestamp()`: Gets current time

## Instrumentation

### Logging and Statistics

- `LOG_LEVEL`: Controls logging verbosity
- `statistics`: Tracks general statistics
- `latencyStatistics`: Tracks order latency
- `slippageStatistics`: Tracks execution slippage

### Observers

- `algorithmObservers`: List of observers
- `register(AlgorithmObserver)`: Registers new observer
- `algorithmNotifier`: Notifies observers of events

## UI Components

- `uiEnabled`: Flag for UI activation
- `startUI()`: Starts user interface
- `setTheme()`: Sets UI theme

## How to Extend

To create a new algorithm:

1. Extend the `Algorithm` class
2. Override abstract methods like `printAlgo()`
3. Implement trading logic in market data handlers
4. Use `sendOrderRequest()` to place orders
5. Configure parameters via `setParameters()`

## Important Implementation Notes

- Thread safety is maintained through synchronization on critical operations
- Orders should be validated before sending
- Position and P&L are automatically tracked
- Market data events drive algorithm execution
- Time management differs between live trading and backtesting

## Examples

Example implementations include:

- Market making algorithms
- Factor investing algorithms
- Reinforcement learning algorithms (`SingleInstrumentRLAlgorithm`)
