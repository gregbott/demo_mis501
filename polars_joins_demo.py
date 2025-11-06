import marimo

__generated_with = "0.10.4"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __(mo):
    mo.md(
        r"""
        # Polars Join Operations: A Comprehensive Guide

        This notebook demonstrates different types of join operations using **Polars**, a blazingly fast DataFrame library.

        We'll work with three related datasets:
        - **Employees**: Information about company employees
        - **Departments**: Information about company departments
        - **Projects**: Information about ongoing projects

        These datasets share common keys that allow us to join them in various ways.
        """
    )
    return


@app.cell
def __():
    import polars as pl
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import Rectangle, FancyBboxPatch
    import numpy as np
    return FancyBboxPatch, Rectangle, mpatches, np, pl, plt


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Dataset 1: Employees

        Our employees table contains information about staff members and their assigned departments.
        """
    )
    return


@app.cell
def __(pl):
    employees = pl.DataFrame({
        "employee_id": [101, 102, 103, 104, 105],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "department_id": [1, 2, 1, 3, 5]  # Note: department_id 5 doesn't exist in departments
    })
    employees
    return (employees,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Dataset 2: Departments

        Our departments table contains department information and their managers.
        """
    )
    return


@app.cell
def __(pl):
    departments = pl.DataFrame({
        "department_id": [1, 2, 3, 4],  # Note: department_id 4 has no employees
        "department_name": ["Engineering", "Sales", "Marketing", "HR"],
        "location": ["NYC", "LA", "Chicago", "NYC"]
    })
    departments
    return (departments,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Dataset 3: Projects

        Our projects table contains ongoing projects assigned to departments.
        """
    )
    return


@app.cell
def __(pl):
    projects = pl.DataFrame({
        "project_id": ["P1", "P2", "P3", "P4"],
        "project_name": ["Website Redesign", "Sales Campaign", "Brand Strategy", "Recruitment Drive"],
        "department_id": [1, 2, 3, 4],
        "budget": [100000, 50000, 75000, 30000]
    })
    projects
    return (projects,)


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 1: INNER JOIN

        An **inner join** returns only the rows where there is a match in **both** tables.

        Think of it as the intersection of two sets - you only get records that exist in both tables.

        **Use Case**: When you only want to work with complete data where relationships exist on both sides.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    inner_join_result = employees.join(
        departments,
        on="department_id",
        how="inner"
    )
    inner_join_result
    return (inner_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: Eve (employee_id 105) is missing because department_id 5 doesn't exist in the departments table. Department 4 (HR) is also missing because no employees are assigned to it.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 2: LEFT JOIN

        A **left join** returns all rows from the left table, and matching rows from the right table. If there's no match, NULL values are filled in for the right table's columns.

        **Use Case**: When you want to keep all records from your primary table, even if they don't have a match.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    left_join_result = employees.join(
        departments,
        on="department_id",
        how="left"
    )
    left_join_result
    return (left_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: Eve appears in the result, but her department_name and location are null because department_id 5 doesn't exist in the departments table.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 3: RIGHT JOIN

        A **right join** is the opposite of a left join - it returns all rows from the right table and matching rows from the left table.

        **Note**: Polars doesn't have a native "right" join, but we can achieve it by swapping the tables and doing a left join, or using "full" join and filtering.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    # Simulate right join by swapping tables
    right_join_result = departments.join(
        employees,
        on="department_id",
        how="left"
    )
    right_join_result
    return (right_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: Department 4 (HR) appears even though no employees are assigned to it. Eve doesn't appear because she's in department 5 which doesn't exist in the departments table.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 4: OUTER JOIN (FULL JOIN)

        An **outer join** (also called full outer join) returns all rows from both tables. Where there's no match, NULL values are filled in.

        **Use Case**: When you want to see all records from both tables, including those that don't have matches.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    outer_join_result = employees.join(
        departments,
        on="department_id",
        how="outer",
        coalesce=True  # Merge the join columns
    )
    outer_join_result
    return (outer_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: We get both Eve (no matching department) AND Department 4 (no matching employees) in our result.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 5: CROSS JOIN

        A **cross join** returns the Cartesian product of both tables - every row from the left table is combined with every row from the right table.

        **Use Case**: When you need to generate all possible combinations of records.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    # For demonstration, let's use smaller datasets
    small_employees = employees.filter(pl.col("employee_id").is_in([101, 102]))
    small_departments = departments.filter(pl.col("department_id").is_in([1, 2]))

    cross_join_result = small_employees.join(
        small_departments,
        how="cross"
    )
    cross_join_result
    return cross_join_result, small_departments, small_employees


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: 2 employees × 2 departments = 4 rows. Each employee is paired with each department, regardless of their actual department_id.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 6: SEMI JOIN

        A **semi join** returns rows from the left table where there is a match in the right table, but doesn't include any columns from the right table.

        **Use Case**: When you want to filter the left table based on existence in the right table.
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    semi_join_result = employees.join(
        departments,
        on="department_id",
        how="semi"
    )
    semi_join_result
    return (semi_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: We get only employees who have matching departments, but no department columns are included. Eve is excluded because department 5 doesn't exist.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Join Type 7: ANTI JOIN

        An **anti join** returns rows from the left table where there is NO match in the right table.

        **Use Case**: When you want to find records that don't have a relationship in another table (orphaned records).
        """
    )
    return


@app.cell
def __(departments, employees, pl):
    anti_join_result = employees.join(
        departments,
        on="department_id",
        how="anti"
    )
    anti_join_result
    return (anti_join_result,)


@app.cell
def __(mo):
    mo.md(
        r"""
        **Notice**: We only get Eve, because she's the only employee whose department_id doesn't exist in the departments table.
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Chaining Joins: Combining All Three Datasets

        In practice, you often need to join multiple tables together. Let's join all three of our datasets.
        """
    )
    return


@app.cell
def __(departments, employees, pl, projects):
    # First join employees with departments
    emp_dept = employees.join(departments, on="department_id", how="inner")

    # Then join with projects
    full_join_result = emp_dept.join(projects, on="department_id", how="left")
    full_join_result
    return emp_dept, full_join_result


@app.cell
def __(mo):
    mo.md(
        r"""
        This gives us a complete view of employees, their departments, and their department's projects!
        """
    )
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Visual Representation of Join Types

        Let's create visual diagrams to understand how different joins work.
        """
    )
    return


@app.cell
def __(FancyBboxPatch, Rectangle, mpatches, plt):
    def create_join_visualization():
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        fig.suptitle('SQL Join Types Visualization', fontsize=16, fontweight='bold')

        join_types = [
            ('INNER JOIN', 'Only matching records'),
            ('LEFT JOIN', 'All left + matching right'),
            ('RIGHT JOIN', 'All right + matching left'),
            ('OUTER JOIN', 'All records from both'),
            ('CROSS JOIN', 'All combinations'),
            ('SEMI JOIN', 'Left where match exists'),
            ('ANTI JOIN', 'Left where no match'),
            ('', '')  # Empty for symmetry
        ]

        for idx, (ax, (join_type, description)) in enumerate(zip(axes.flat, join_types)):
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis('off')

            if join_type == '':
                continue

            # Draw two circles representing tables
            left_circle = plt.Circle((3, 3), 1.5, color='#3498db', alpha=0.3, label='Left Table')
            right_circle = plt.Circle((7, 3), 1.5, color='#e74c3c', alpha=0.3, label='Right Table')

            if join_type == 'INNER JOIN':
                # Only intersection
                ax.add_patch(plt.Circle((5, 3), 0.75, color='#9b59b6', alpha=0.7))
                ax.add_patch(left_circle)
                ax.add_patch(right_circle)

            elif join_type == 'LEFT JOIN':
                # Full left + intersection
                ax.add_patch(plt.Circle((3, 3), 1.5, color='#3498db', alpha=0.7))
                ax.add_patch(plt.Circle((5, 3), 0.75, color='#9b59b6', alpha=0.9))
                ax.add_patch(right_circle)

            elif join_type == 'RIGHT JOIN':
                # Full right + intersection
                ax.add_patch(left_circle)
                ax.add_patch(plt.Circle((7, 3), 1.5, color='#e74c3c', alpha=0.7))
                ax.add_patch(plt.Circle((5, 3), 0.75, color='#9b59b6', alpha=0.9))

            elif join_type == 'OUTER JOIN':
                # Both circles fully filled
                ax.add_patch(plt.Circle((3, 3), 1.5, color='#3498db', alpha=0.7))
                ax.add_patch(plt.Circle((7, 3), 1.5, color='#e74c3c', alpha=0.7))
                ax.add_patch(plt.Circle((5, 3), 0.75, color='#9b59b6', alpha=0.9))

            elif join_type == 'CROSS JOIN':
                # Show grid pattern
                ax.add_patch(Rectangle((2, 2), 2, 2, color='#3498db', alpha=0.5))
                ax.add_patch(Rectangle((6, 2), 2, 2, color='#e74c3c', alpha=0.5))
                ax.plot([4, 6], [3, 3], 'k-', linewidth=2)
                ax.plot([4, 6], [3, 3], 'ko', markersize=8)
                ax.text(5, 1, '×', fontsize=24, ha='center')

            elif join_type == 'SEMI JOIN':
                # Left circle only (where match exists)
                overlap_left = plt.Circle((3.5, 3), 1, color='#3498db', alpha=0.7)
                ax.add_patch(left_circle)
                ax.add_patch(right_circle)
                ax.add_patch(overlap_left)

            elif join_type == 'ANTI JOIN':
                # Left circle only (where no match)
                left_only = plt.Circle((2.5, 3), 1, color='#3498db', alpha=0.7)
                ax.add_patch(left_circle)
                ax.add_patch(right_circle)
                ax.add_patch(left_only)

            # Add title and description
            ax.text(5, 5.3, join_type, fontsize=12, fontweight='bold', ha='center')
            ax.text(5, 0.7, description, fontsize=9, ha='center', style='italic')

            # Add labels
            ax.text(3, 3, 'L', fontsize=12, ha='center', va='center', fontweight='bold')
            ax.text(7, 3, 'R', fontsize=12, ha='center', va='center', fontweight='bold')

        plt.tight_layout()
        return fig

    join_viz = create_join_visualization()
    join_viz
    return create_join_visualization, join_viz


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Data Flow Visualization: Tracking Record Counts

        Let's see how many records survive each type of join.
        """
    )
    return


@app.cell
def __(departments, employees, np, pl, plt):
    def create_record_count_chart():
        # Calculate record counts for each join type
        join_counts = {
            'Original\nEmployees': len(employees),
            'Original\nDepartments': len(departments),
            'Inner\nJoin': len(employees.join(departments, on="department_id", how="inner")),
            'Left\nJoin': len(employees.join(departments, on="department_id", how="left")),
            'Right\nJoin': len(departments.join(employees, on="department_id", how="left")),
            'Outer\nJoin': len(employees.join(departments, on="department_id", how="outer", coalesce=True)),
            'Semi\nJoin': len(employees.join(departments, on="department_id", how="semi")),
            'Anti\nJoin': len(employees.join(departments, on="department_id", how="anti")),
        }

        fig, ax = plt.subplots(figsize=(12, 6))

        labels = list(join_counts.keys())
        counts = list(join_counts.values())
        colors = ['#3498db', '#e74c3c', '#9b59b6', '#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#e67e22']

        bars = ax.bar(labels, counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

        ax.set_ylabel('Number of Records', fontsize=12, fontweight='bold')
        ax.set_xlabel('Join Type', fontsize=12, fontweight='bold')
        ax.set_title('Record Counts by Join Type\n(Employees ⋈ Departments)', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(counts) + 1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    record_count_chart = create_record_count_chart()
    record_count_chart
    return create_record_count_chart, record_count_chart


@app.cell
def __(mo):
    mo.md(
        r"""
        ---
        ## Summary Table: When to Use Each Join Type

        | Join Type | Returns | Use Case |
        |-----------|---------|----------|
        | **INNER** | Only matching records from both tables | When you need complete data on both sides |
        | **LEFT** | All from left + matching from right | Keep all primary records, add related info when available |
        | **RIGHT** | All from right + matching from left | Keep all secondary records, add related info when available |
        | **OUTER** | All records from both tables | When you need everything, matches or not |
        | **CROSS** | Cartesian product (all combinations) | Generate all possible pairings |
        | **SEMI** | Left records that have a match (no right columns) | Filter based on existence in another table |
        | **ANTI** | Left records that DON'T have a match | Find orphaned/unmatched records |

        ---

        ## Key Takeaways

        1. **Join keys matter**: Make sure your join columns contain the right data types and matching values
        2. **NULL handling**: Outer joins introduce NULLs - be prepared to handle them
        3. **Performance**: Inner and semi joins are typically fastest; cross joins can explode in size
        4. **Polars specifics**: Use `coalesce=True` in outer joins to merge join columns
        5. **Chain carefully**: Order matters when joining multiple tables

        **Pro tip**: Always check your record counts before and after joining to ensure you're getting expected results!
        """
    )
    return


if __name__ == "__main__":
    app.run()
