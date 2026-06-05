# 使用 pre commit hook 来 format 代码

手动 format 代码比较费事情， 比较主流的做法是使用 pre commit hook

```
pip install pre-commit
git config --global --unset core.hooksPath
pre-commit install
```
