cd gpatch
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd gdataset
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd megatron_datasets
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd tasks
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd tools
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd tests
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd examples
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

isort gdataset/data_loader

isort gpatch_v4

cd gpatch_v4
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..

cd tests/test_gpatch_v4
yapf -i --style="{ based_on_style: facebook, column_limit: 100 }" `find -name '*.py'`
cd ..
