class ComparisonResult:
    """
    A class to represent comparison results between two lists.
    Each comparison result includes four integer attributes:
    top_1, top_2, top_3, and any_position.
    """

    def __init__(self, top_1=0, top_2=0, top_3=0, any_position=0):
        self.top_1 = int(top_1)
        self.top_2 = int(top_2)
        self.top_3 = int(top_3)
        self.any_position = int(any_position)

    def __add__(self, other):
        """
        Allows addition of two ComparisonResult objects to aggregate results.
        """
        if not isinstance(other, ComparisonResult):
            raise TypeError("Can only add ComparisonResult instances")

        return ComparisonResult(
            top_1=self.top_1 + other.top_1,
            top_2=self.top_2 + other.top_2,
            top_3=self.top_3 + other.top_3,
            any_position=self.any_position + other.any_position
        )

    def __repr__(self):
        return (f"ComparisonResult(top_1={self.top_1}, "
                f"top_2={self.top_2}, "
                f"top_3={self.top_3}, "
                f"any_position={self.any_position})")

    @staticmethod
    def top_1_hit():
        """
        Creates a ComparisonResult where top_1 hit is counted.
        """
        return ComparisonResult(top_1=1, top_2=1, top_3=1, any_position=1)

    @staticmethod
    def top_2_hit():
        """
        Creates a ComparisonResult where top_2 hit is counted.
        """
        return ComparisonResult(top_1=0, top_2=1, top_3=1, any_position=1)

    @staticmethod
    def top_3_hit():
        """
        Creates a ComparisonResult where top_3 hit is counted.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=1, any_position=1)

    @staticmethod
    def any_position_hit():
        """
        Creates a ComparisonResult where any position hit is counted,
        but not in top 1-3.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=0, any_position=1)

    @staticmethod
    def no_hit():
        """
        Creates a ComparisonResult where there is no hit.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=0, any_position=0)


if __name__ == '__main__':
    # Test cases using static methods
    cr1 = ComparisonResult.top_1_hit()
    print("Test Case 1 (Top 1 Hit):", cr1)

    cr2 = ComparisonResult.top_2_hit()
    print("Test Case 2 (Top 2 Hit):", cr2)

    cr3 = ComparisonResult.top_3_hit()
    print("Test Case 3 (Top 3 Hit):", cr3)

    cr4 = ComparisonResult.any_position_hit()
    print("Test Case 4 (Any Position Hit):", cr4)

    cr5 = ComparisonResult.no_hit()
    print("Test Case 5 (No Hit):", cr5)

    # Aggregating results
    total_result = cr1 + cr2 + cr3 + cr4 + cr5
    print("Aggregated Result:", total_result)
