# Market Making Algorithms Documentation

## Overview

This document provides detailed information about the market making algorithms implemented in the Lambda Investing
framework. These algorithms place limit orders on both sides of the order book to provide liquidity and capture the
bid-ask spread.

## 1. AvellanedaStoikov Algorithm

### Theoretical Background

The Avellaneda-Stoikov algorithm is based on the seminal paper "High-frequency trading in a limit order book" by Marco
Avellaneda and Sasha Stoikov (2008). It models market making as an inventory management problem with a risk aversion
parameter.

### Implementation Details

The algorithm dynamically adjusts bid and ask quotes based on:

- Mid-price of the instrument
- Current inventory position
- Time remaining until trading day end
- Market volatility
- Risk aversion parameter

### Key Parameters

- `riskAversion`: Controls how aggressively the algorithm manages inventory risk (higher values = tighter spreads when
  inventory deviates from zero)
- `quantity`: Size of each order placed
- `windowLength`: Time window for volatility calculation (in seconds)
- `gamma`: Mean-reversion parameter for mid-price
- `kappa`: Spread dampening factor
- `maxSpreadCurrency`: Maximum allowed spread in currency units
- `minSpreadCurrency`: Minimum allowed spread in currency units
- `inventoryLimit`: Maximum allowed inventory position

### Trading Logic

1. Calculates optimal bid and ask prices using the Avellaneda-Stoikov formula:
   ```
   r = risk aversion
   σ = volatility
   q = current inventory
   T = time remaining
   S = mid price
   
   reservation_price = S - q * r * σ² * (T)
   optimal_spread = 2/r * log(1 + r/k)
   
   bid = reservation_price - optimal_spread/2
   ask = reservation_price + optimal_spread/2
   ```

2. Applies spread constraints (min/max spread)
3. Places limit orders at calculated prices
4. Adjusts prices as market conditions change
5. Manages inventory through price adjustments

### Code Structure

- `onDepthUpdate()`: Main entry point for market data processing
- `calculateReservationPrice()`: Computes reservation price based on inventory
- `calculateOptimalSpread()`: Determines optimal spread
- `updateOrders()`: Places or updates orders in the book

## 2. AlphaAvellanedaStoikov Algorithm

### Overview

An extension of the standard Avellaneda-Stoikov algorithm that incorporates alpha signals to predict short-term price
movements.

### Key Differences from AvellanedaStoikov

- Includes an alpha component that shifts the reservation price
- Can use external signals or internally generated predictions
- More aggressive in positioning when alpha signals are strong

### Additional Parameters

- `alphaSignalName`: Name of the alpha signal to use
- `alphaScaling`: Scaling factor for alpha signal impact
- `alphaSignalThreshold`: Threshold for considering alpha signals
- `alphaHalfLife`: Half-life for alpha signal decay

### Trading Logic

1. Retrieves alpha signal value
2. Adjusts reservation price based on alpha:
   ```
   adjusted_reservation_price = reservation_price + alpha_signal * alpha_scaling
   ```
3. Uses adjusted reservation price in spread calculations
4. Places orders with asymmetric spreads based on alpha direction

### Signal Integration

The algorithm can integrate with various alpha signals:

- Technical indicators
- Order flow imbalance
- News sentiment
- Statistical arbitrage signals

## 3. AlphaConstantSpread Algorithm

### Overview

A market making algorithm that uses a constant base spread but adjusts quote positioning based on alpha signals.

### Key Features

- Uses constant spread values as the baseline
- Shifts both quotes in the direction of the alpha signal
- Maintains asymmetric quotes when alpha signal is strong
- Simpler than Avellaneda-Stoikov but still responsive to directional signals

### Key Parameters

- `baseSpread`: The default spread in ticks or currency
- `alphaSignalName`: Name of the alpha signal to use
- `alphaScaling`: How strongly alpha affects quote positioning
- `spreadMultiplierUp`: Increases spread when inventory exceeds limits
- `spreadMultiplierDown`: Decreases spread when market conditions are favorable
- `maxPositionThreshold`: Position size that triggers spread adjustments

### Trading Logic

1. Calculates base spread around the mid price
2. Retrieves and scales alpha signal
3. Shifts both quotes in direction of alpha:
   ```
   shift_amount = alpha_signal * alpha_scaling
   bid = mid_price - (base_spread/2) + shift_amount
   ask = mid_price + (base_spread/2) + shift_amount
   ```
4. Adjusts spread based on current inventory position
5. Places or updates orders at calculated prices

### Risk Management

- Widens spread when inventory approaches limits
- Can stop quoting one side when inventory limit is reached
- Implements "skewing" by adjusting bid/ask quantities asymmetrically

## 4. ConstantSpreadAlgorithm

### Overview

The simplest market making algorithm that maintains a fixed spread around the mid price without considering alpha
signals or sophisticated inventory management.

### Key Features

- Fixed spread around mid price
- Basic inventory controls
- Computationally efficient
- Good baseline for comparing more complex strategies

### Key Parameters

- `spreadTicks`: Fixed spread in number of price ticks
- `quantity`: Size of each order
- `maxPosition`: Maximum allowed position (absolute value)
- `firstHour`: First hour of trading (time filtering)
- `lastHour`: Last hour of trading (time filtering)
- `tickSize`: Minimum price increment

### Trading Logic

1. Calculates bid and ask prices using fixed spread:
   ```
   bid = mid_price - (spread_ticks * tick_size / 2)
   ask = mid_price + (spread_ticks * tick_size / 2)
   ```
2. Checks if current position allows placing orders on both sides
3. Places or updates orders at calculated prices
4. Cancels orders on one side if position limit is reached

### Position Management

- Simple threshold-based position limits
- Does not dynamically adjust prices based on inventory
- Can skip placing orders on one side when position limit is reached

### Use Cases

- Market making in low-volatility instruments
- Baseline comparison for more complex algorithms
- Educational purposes and algorithm development
- Markets where sophisticated pricing is not necessary

## Performance Comparison

| Algorithm               | Complexity | Inventory Management | Alpha Integration | Computational Load | Best Use Case                            |
|-------------------------|------------|----------------------|-------------------|--------------------|------------------------------------------|
| AvellanedaStoikov       | High       | Sophisticated        | No                | Medium             | High-frequency trading in liquid markets |
| AlphaAvellanedaStoikov  | Very High  | Sophisticated        | Yes               | High               | Directional trading with market making   |
| AlphaConstantSpread     | Medium     | Basic                | Yes               | Low                | Markets with clear signals               |
| ConstantSpreadAlgorithm | Low        | Minimal              | No                | Very Low           | Stable, low-volatility markets           |

## Implementation Notes

### Common Methods Across Algorithms

All market making algorithms share these key methods:

- `onDepthUpdate()`: Processes order book updates
- `calculateQuotes()`: Determines bid/ask prices
- `updateOrders()`: Places or modifies orders
- `checkInventory()`: Manages position limits
- `cancelAllOrders()`: Cancels all active orders
- `handleFill()`: Processes execution reports

### Integration with Framework

- All algorithms extend the base `Algorithm` class
- They implement the `MarketDataListener` interface
- Position tracking is handled by the base class
- P&L calculation is automatic
- Parameter setting is done via JSON configuration

### Backtesting Considerations

- Set `riskAversion` higher for more conservative strategies
- Test different `quantity` values for varying market depths
- Analyze performance across different volatility regimes
- Compare to `ConstantSpreadAlgorithm` as a baseline
- Track inventory evolution for risk management assessment

## Example Configuration

```json
{
  "algorithm": {
    "algorithmName": "AvellanedaStoikov",
    "parameters": {
      "riskAversion": 0.00006,
      "quantity": 0.001,
      "windowLength": 60,
      "gamma": 0.1,
      "kappa": 1.5,
      "maxSpreadCurrency": 20.0,
      "minSpreadCurrency": 0.5,
      "inventoryLimit": 0.01,
      "firstHour": 7.0,
      "lastHour": 19.0
    }
  }
}
```
