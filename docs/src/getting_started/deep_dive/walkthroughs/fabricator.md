# Fabricator: Declarative Synthetic Data Generation

## Why Use Fabricator?

When developing and testing data pipelines, having realistic test data is crucial. However, manually creating test data is:
- Time-consuming and error-prone
- Hard to maintain as schemas evolve
- Difficult to make reproducible
- Often lacks the complexity of real-world data

The Fabricator module solves these challenges by providing a declarative way to generate synthetic data that:
- Closely mimics real-world data patterns
- Maintains referential integrity across tables
- Generates deterministically when seeded
- Scales easily as schemas grow
- Keeps test data generation code clean and maintainable

## What Does It Do?

Fabricator takes a YAML configuration that describes your desired data structure and generates pandas DataFrames accordingly. Key features include:

1. **Rich Data Generation Options**
   - Unique IDs with customizable formats
   - Realistic fake data (names, addresses, etc.) via Faker
   - Statistical distributions via NumPy
   - Custom arrays and lists
   - Date sequences and ranges
   - Value mapping and transformations

2. **Complex Relationships**
   - Reference columns from other tables
   - Cross products for exhaustive combinations
   - Row-wise and column-wise operations
   - Weighted random sampling
   - Deterministic value mapping

3. **Data Quality Controls**
   - Configurable null value injection
   - Type casting and validation
   - Size control (up/down sampling)
   - Seeding for reproducibility
   - Error handling and logging

## How to Use It

### Basic Example

Let's start with a simple example generating patient data:

```yaml
patients:
    num_rows: 100
    columns:
        id:
            type: generate_unique_id
            prefix: PAT_
            id_length: 8
        name:
            type: faker
            provider: name
        admission_date:
            type: generate_dates
            start_dt: 2023-01-01
            end_dt: 2023-12-31
            freq: B  # Business days
```

This generates a DataFrame with:
- 100 rows
- Patient IDs like 'PAT_00000123'
- Realistic names
- Random admission dates on business days in 2023

### Advanced Features

#### 1. Column References and Dependencies

You can reference columns from the same or other tables:

```yaml
encounters:
    num_rows: 500
    columns:
        id:
            type: generate_unique_id
            prefix: ENC_
        patient_id:
            type: copy_column
            source_column: patients.id
            sample:
                num_rows: "@encounters.num_rows"
        visit_date:
            type: row_apply
            input_columns: [patients.admission_date]
            row_func: "lambda x: x + datetime.timedelta(days=random.randint(1, 30))"
```

#### 2. Complex Data Generation

Generate arrays, weighted choices, and statistical distributions:

```yaml
diagnoses:
    num_rows: 200
    columns:
        codes:
            type: generate_random_arrays
            sample_values: ["E11.9", "I10", "J45.909", "F41.1"]
            min_length: 1
            max_length: 3
            delimiter: "|"  # Results in strings like "E11.9|I10"
        severity:
            type: generate_values
            sample_values:
                mild: 0.5    # 50% chance
                moderate: 0.3 # 30% chance
                severe: 0.2   # 20% chance
        lab_value:
            type: numpy_random
            distribution: normal
            loc: 100    # mean
            scale: 15   # standard deviation
```

#### 3. Data Quality Controls

Control nulls and types:

```yaml
demographics:
    columns:
        email:
            type: faker
            provider: email
            inject_nulls:
                probability: 0.1
                value: "NOEMAIL"
        age:
            type: numpy_random
            distribution: normal
            loc: 45
            scale: 15
            dtype: Int64  # Nullable integer type
```

### Real-World Example

Here's a more complex example showing how Fabricator handles real-world data patterns:

```yaml
rtx_kg2:  # Knowledge Graph Data
    nodes:
        num_rows: 500
        columns:
            id:
                type: generate_unique_id
                prefix: "RTX:"
            name:
                type: generate_unique_id
                prefix: name_
                inject_nulls:
                    probability: 0.2
            category:
                type: generate_values
                sample_values:
                    - biolink:Drug
                    - biolink:Disease
                    - biolink:Gene
            all_categories:
                type: row_apply
                input_columns: ["nodes.category"]
                row_func: matrix.pipelines.fabricator.generators.get_ancestors_for_category_delimited
                row_func_kwargs:
                    delimiter: "|"
            equivalent_curies:
                type: generate_random_arrays
                delimiter: "|"
                sample_values:
                    - "CHEMBL:CHEMBL25"
                    - "DRUGBANK:DB00316"
                    - "MESH:D009369"

    edges:
        num_rows: 2000
        columns:
            subject:
                type: copy_column
                source_column: "nodes.id"
                sample:
                    num_rows: "@edges.num_rows"
                    seed: 590590
            object:
                type: copy_column
                source_column: "nodes.id"
                sample:
                    num_rows: "@edges.num_rows"
                    seed: 49494
            predicate:
                type: generate_values
                sample_values:
                    - biolink:treats
                    - biolink:interacts_with
                    - biolink:affects
```

### Best Practices

1. **Use Seeds for Reproducibility**
   - Set a global seed when initializing MockDataGenerator
   - Use specific seeds for critical columns
   - Document seed values in comments

2. **Structure Your Configuration**
   - Group related tables under namespaces
   - Order tables by dependency (referenced tables first)
   - Use clear, consistent naming

3. **Handle Data Quality**
   - Configure appropriate null rates
   - Set explicit dtypes for important columns
   - Use validation where needed

4. **Performance Considerations**
   - Use column_apply for operations needing full column context
   - Use row_apply for independent row operations
   - Consider chunking for very large datasets

### Common Patterns

1. **Stable References**
   ```yaml
   # Generate stable mappings using hash_map
   territory:
       type: hash_map
       input_column: postal_code
       buckets: ["North", "South", "East", "West"]
   ```

2. **Derived Values**
   ```yaml
   # Calculate values based on other columns
   bmi:
       type: row_apply
       input_columns: [weight_kg, height_m]
       row_func: "lambda w, h: round(w / (h * h), 1)"
   ```

3. **Complex Combinations**
   ```yaml
   # Generate all possible combinations
   product_regions:
       type: column_apply
       input_columns: [products.id, regions.id]
       column_func: cross_product
       check_all_inputs_same_length: false
   ```

### Troubleshooting

Common issues and solutions:

**Column Reference Errors:**

- Ensure referenced tables/columns are generated first
- Check namespace prefixes if using multiple namespaces
- Verify column names match exactly

**Type Mismatches:**

- Set explicit dtypes when needed
- Check format of date strings
- Use appropriate nullable types (Int64, string, etc.)

**Size Mismatches:**

- Use resize=True when needed
- Check num_rows references
- Verify sample configurations

**Seeding Issues:**

- Set global seed for overall reproducibility
- Use column-specific seeds for fine control
- Document seed values used

## Further Reading

- [Faker Providers Documentation](https://faker.readthedocs.io/en/master/providers.html)
- [NumPy Random Generator](https://numpy.org/doc/stable/reference/random/generator.html)
- [Pandas Data Types](https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes)