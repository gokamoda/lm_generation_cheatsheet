def check_power_of_2(x: int) -> bool:
    return x != 0 and (x & (x - 1)) == 0
