class ComparisonResult:
    """
    A class to represent comparison results between two lists.
    Each comparison result includes four integer attributes:
    top_1, top_2, top_3, and top_any.
    """

    def __init__(self, top_1=0, top_2=0, top_3=0, top_any=0, n_samples=1):
        self.top_1 = int(top_1)
        self.top_2 = int(top_2)
        self.top_3 = int(top_3)
        self.top_any = int(top_any)
        self.n_samples = int(n_samples)

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
            top_any=self.top_any + other.top_any,
            n_samples=self.n_samples + other.n_samples
        )

    def __lt__(self, other):
        """
        Less-than comparison for ComparisonResult objects.
        Prioritizes top_1_rate, then top_2_rate, top_3_rate, and top_any_rate.
        """
        if not isinstance(other, ComparisonResult):
            return NotImplemented

        # Compare rates in order of importance
        return (
            (self.get_top_1_rate(), self.get_top_2_rate(),
             self.get_top_3_rate(), self.get_top_any_rate())
            < (other.get_top_1_rate(), other.get_top_2_rate(), other.get_top_3_rate(), other.get_top_any_rate())
        )

    def __gt__(self, other):
        """
        Greater-than comparison for ComparisonResult objects.
        Prioritizes top_1_rate, then top_2_rate, top_3_rate, and top_any_rate.
        """
        if not isinstance(other, ComparisonResult):
            return NotImplemented

        # Compare rates in order of importance
        return (
            (self.get_top_1_rate(), self.get_top_2_rate(),
             self.get_top_3_rate(), self.get_top_any_rate())
            > (other.get_top_1_rate(), other.get_top_2_rate(), other.get_top_3_rate(), other.get_top_any_rate())
        )

    def __repr__(self):
        return (f"ComparisonResult(top_1={self.top_1}, "
                f"top_2={self.top_2}, "
                f"top_3={self.top_3}, "
                f"top_any={self.top_any}, "
                f"n_samples={self.n_samples})")

    def get_top_1_rate(self):
        return self.top_1 / self.n_samples

    def get_top_2_rate(self):
        return self.top_2 / self.n_samples

    def get_top_3_rate(self):
        return self.top_3 / self.n_samples

    def get_top_any_rate(self):
        return self.top_any / self.n_samples

    def get_n_samples(self):
        return self.n_samples

    @staticmethod
    def top_1_hit():
        """
        Creates a ComparisonResult where top_1 hit is counted.
        A hit in top 1 is also considered a hit in top 2, top 3, and top_any.
        """
        return ComparisonResult(top_1=1, top_2=1, top_3=1, top_any=1)

    @staticmethod
    def top_2_hit():
        """
        Creates a ComparisonResult where top_2 hit is counted.
        A hit in top 2 is also considered a hit in top 3 and top_any.
        """
        return ComparisonResult(top_1=0, top_2=1, top_3=1, top_any=1)

    @staticmethod
    def top_3_hit():
        """
        Creates a ComparisonResult where top_3 hit is counted.
        A hit in top 3 is also considered a hit in top_any.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=1, top_any=1)

    @staticmethod
    def top_any_hit():
        """
        Creates a ComparisonResult where a hit in any position is counted,
        but not in top 1-3.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=0, top_any=1)

    @staticmethod
    def no_hit():
        """
        Creates a ComparisonResult where there is no hit.
        """
        return ComparisonResult(top_1=0, top_2=0, top_3=0, top_any=0)


if __name__ == '__main__':
    # Test cases using static methods
    cr1 = ComparisonResult.top_1_hit()
    print("Test Case 1 (Top 1 Hit):", cr1)

    cr2 = ComparisonResult.top_2_hit()
    print("Test Case 2 (Top 2 Hit):", cr2)

    cr3 = ComparisonResult.top_3_hit()
    print("Test Case 3 (Top 3 Hit):", cr3)

    cr4 = ComparisonResult.top_any_hit()
    print("Test Case 4 (Top Any Hit):", cr4)

    cr5 = ComparisonResult.no_hit()
    print("Test Case 5 (No Hit):", cr5)

    # Aggregating results
    total_result = cr1 + cr2 + cr3 + cr4 + cr5
    print("Aggregated Result:", total_result)
