# Skill: code_review

## Purpose
Perform comprehensive code validation based on system architecture and implementation rules.

## When to use
- code generation
- debugging
- pipeline validation

## Step 1: Architecture Validation (HIGH PRIORITY)
Use system_architecture to check:
- whether the pipeline follows: data → forward → loss → backward
- whether modules interact correctly
- whether registry system is correctly used
- whether any step is missing or inconsistent
- whether cfg.xxx access paths match the YParams loaded section and YAML nesting

## Step 2: Implementation Checks
Use the following sub-skills:

- bug_check → general bug detection
- dimension_check → tensor shape validation
- device_adaptation → device consistency
- onescience_reuse → module reuse validation

## Output
1. Architecture Issues (if any)
2. Implementation Issues
3. Root Causes
4. Fix Suggestions
5. Risk Level