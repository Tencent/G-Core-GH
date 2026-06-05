# 打包

[腾讯内部 pip 仓库](https://mirrors.tencent.com/#/private/pypi/detail?repo_id=155&project_name=gcore&search_label=package_name&search_value=gcore&page_num=1)

上传
```bash
python3 setup.py bdist_wheel
twine upload dist/* --repository-url https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple --username ${1} --password ${2}
```

安装
```bash
pip3 install gcore --index-url=https://mirrors.tencent.com/repository/pypi/tencent_pypi/simple
```
