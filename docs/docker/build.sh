readonly space=wepsdl
readonly name=docserver

function build() {
  tag=`date +%s`
  docker build --network=host -f Dockerfile -t "$space/$name:$tag" .

  hub='weregistry.woa.com'
  docker tag "$space/$name:$tag" "$hub/$space/$name:$tag"
  docker push "$hub/$space/$name:$tag"
  docker image remove "$hub/$space/$name:$tag"
}

rm -rf html
cp -r ../build/html .
build
rm -rf html
