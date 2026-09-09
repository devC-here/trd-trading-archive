prices = [7,1,5,3,6,4]
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0

        diff = prices[1] - prices[0]

        if diff < 0:
            return 0

        return diff
