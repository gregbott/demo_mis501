import marimo

__generated_with = "0.16.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    # Dictionary Practice Exercises
    # Complete each exercise below

    # Exercise 1: Basic Dictionary Creation and Access
    # Create a dictionary called 'student' with keys: 'name', 'age', 'grade', 'school'
    # Then print the student's name and grade

    student = {
        # Your code here
    }

    # Print the name and grade
    # Your code here


    # Exercise 2: Modifying Dictionaries
    # Given this inventory dictionary, add a new item 'banana': 12
    # Update 'apple' quantity to 25
    # Remove 'orange' from the inventory

    inventory = {'apple': 15, 'orange': 8, 'grape': 20}

    # Your code here

    print(inventory)


    # Exercise 3: Dictionary Comprehension - Squares
    # Use a dictionary comprehension to create a dictionary where:
    # - Keys are numbers from 1 to 10
    # - Values are the squares of those numbers
    # Example: {1: 1, 2: 4, 3: 9, ...}

    squares = {}  # Replace with dictionary comprehension

    print(squares)


    # Exercise 4: Iterating Through Dictionaries
    # Given this prices dictionary, calculate and print the total cost
    # Print each item with its price in the format: "Item: $price"

    prices = {'laptop': 899, 'mouse': 25, 'keyboard': 75, 'monitor': 299}

    # Your code here


    # Exercise 5: Dictionary Comprehension - Filtering
    # Use a dictionary comprehension to create a new dictionary from 'scores'
    # that only includes students who scored 70 or above

    scores = {'Alice': 85, 'Bob': 62, 'Charlie': 78, 'Diana': 91, 'Eve': 58}

    passing_scores = {}  # Replace with dictionary comprehension

    print(passing_scores)


    # SOLUTIONS (uncomment to check your answers)
    """
    # Exercise 1 Solution:
    student = {
        'name': 'Alex',
        'age': 16,
        'grade': 'A',
        'school': 'Central High'
    }
    print(f"Name: {student['name']}")
    print(f"Grade: {student['grade']}")

    # Exercise 2 Solution:
    inventory['banana'] = 12
    inventory['apple'] = 25
    del inventory['orange']
    # or: inventory.pop('orange')

    # Exercise 3 Solution:
    squares = {num: num**2 for num in range(1, 11)}

    # Exercise 4 Solution:
    total = 0
    for item, price in prices.items():
        print(f"{item}: ${price}")
        total += price
    print(f"Total: ${total}")

    # Exercise 5 Solution:
    passing_scores = {name: score for name, score in scores.items() if score >= 70}
    """
    return


if __name__ == "__main__":
    app.run()
