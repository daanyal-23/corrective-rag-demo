# TODO: Fix GradeDocuments Function Call Error

## Steps to Complete
- [x] Update `src/tools/rag_resources.py`: Modify the retrieval grader to use explicit JSON output in the prompt and remove structured output setup.
- [x] Update `src/nodes/grade_node.py`: Adjust score extraction to parse JSON response manually.
- [x] Test the changes by running the application to verify the error is resolved.
