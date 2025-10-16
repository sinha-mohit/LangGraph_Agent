def binary_search(arr, target):
    low = 0
    high = len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1  # Target not found

# Example usage
if __name__ == '__main__':
    my_list = [2, 5, 8, 12, 16, 23, 38, 56, 72, 91]
    target_value = 23
    result = binary_search(my_list, target_value)

    if result != -1:
        print(f"Element {target_value} is present at index {result}")
    else:
        print(f"Element {target_value} is not present in the list")
