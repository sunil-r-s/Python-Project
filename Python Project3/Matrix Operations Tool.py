import numpy as np

# ----Step 1: Input matrices from the user----
print("WELCOM TO THE MATRIX OPERATIONS TOOL!\n")
rows = int(input("Enter the number of rows in the first matrix: "))
cols = int(input("Enter the number of columns in the first matrix: "))

a = np.array(list(map(int, (input("\nEnter the first matrix: ").split())))).reshape(rows, cols)
b = np.array(list(map(int, (input("Enter the second matrix: ").split())))).reshape(rows, cols)

# ----Step 2: Menu-driven loop----
while True:
    print("\n========== Matrix Operations Menu ==========")
    print("1. Addition")
    print("2. Subtraction")
    print("3. Multiplication")
    print("4. Division")
    print("5. Transpose (First Matrix)")
    print("6. Determinant (First Matrix)")
    print("7. Inverse (First Matrix)")
    print("8. Exit")
    print("============================================\n")

    choice = input("Enter your choice (1-8): ")

    print("\n----------- Result -----------")
    if choice == "1":
        print("Matrix A:\n", a)
        print("Matrix B:\n", b)
        print("Addition:\n", a + b)

    elif choice == "2":
        print("Subtraction (A - B):\n", a - b)

    elif choice == "3":
        print("Multiplication:\n", a * b)  # element-wise

    elif choice == "4":
        print("Division (A / B):\n", a / b)

    elif choice == "5":
        print("Transpose of Matrix A:\n", a.T)

    elif choice == "6":
        if a.shape[0] == a.shape[1]:
            print("Determinant of Matrix A:", np.linalg.det(a))
        else:
            print("Error: Determinant only for square matrices.")

    elif choice == "7":
        if a.shape[0] == a.shape[1] and np.linalg.det(a) != 0:
            print("Inverse of Matrix A:\n", np.linalg.inv(a))
        else:
            print("Error: Inverse not possible (non-square or singular matrix).")

    elif choice == "8":
        print("Exiting program. Goodbye!")
        break

    else:
        print("Invalid choice! Please try again.")

    print("\n------------------------------")