# Usage: make-pub.sh [append|overwrite]
#   append (default): base on txpub/public, apply diff as a single new commit on top
#   overwrite: discard public history, force-push a single fresh commit
MODE="${1:-append}"
if [[ "$MODE" != "append" && "$MODE" != "overwrite" ]]; then
    echo "Usage: $0 [append|overwrite]" >&2
    exit 1
fi

# make branch
git branch -D prepare_public
git branch -D public
git checkout -b prepare_public

# clean
git rm -rf examples
git rm -rf tests/test_gpatch_v3
git rm -rf `find -type d -name 'priv'`
git rm `find -name '*-priv.*'`
git rm `find -name '*_priv.*'`
git rm -rf `find tasks -mindepth 1 -maxdepth 1 -type d | grep -v 'tasks/math_rl_v4'`
git rm -rf mpatch
git rm -rf gpatch
git rm -rf debug
git rm -rf `find gpatch_v4/models -maxdepth 1 -type d -name 'oteam*'`
git rm -rf `find gpatch_v4/models -maxdepth 1 -type d -name 'weclip*'`
git rm -rf `find gpatch_v4/models -maxdepth 1 -type d -name 'welm*'`
git rm -rf `find gpatch_v4/models -maxdepth 1 -type d -name 'wegen*'`
git rm -f gpatch_v4/extended_pipeline/pipeline_oteam4_3.py gpatch_v4/extended_pipeline/pipeline_oteam4_4.py
git rm -f gpatch_v4/extended_model/welm_v4.py
sed -i 's/WANDB_BASE_URL=.*/WANDB_BASE_URL=/g' `find -name '*.sh' | grep -v make-pub.sh`
sed -i 's/WANDB_API_KEY=.*/WANDB_API_KEY=/g' `find -name '*.sh' | grep -v make-pub.sh`

# Remove report: blocks from YAML files (wandb credentials)
python3 -c "
import re, os, glob

files = []
for p in ['**/*.yaml', '**/*.yml']:
    for f in glob.glob(p, recursive=True):
        parts = f.split(os.sep)
        if 'hf-hub' not in parts:
            files.append(f)

for f in files:
    try:
        with open(f) as fh:
            lines = fh.readlines()
    except (PermissionError, IOError):
        continue

    new_lines = []
    skip_indent = None
    for line in lines:
        stripped = line.rstrip('\n').rstrip('\r')
        if skip_indent is not None:
            if stripped == '' or stripped.lstrip().startswith('#'):
                continue
            cur_indent = len(re.match(r'^(\s*)', line).group(1))
            if cur_indent <= skip_indent:
                skip_indent = None
                new_lines.append(line)
            continue

        m = re.match(r'^(\s*)report:\s*', line)
        if m:
            # Skip report: key and all its children (more-indented lines)
            skip_indent = len(m.group(1))
            # If report: has inline value (report: wandb), just skip this line
            # and stop skipping immediately since there are no children.
            inline_val = line[m.end():].strip()
            if inline_val and not inline_val.startswith('#'):
                skip_indent = None  # single-line report: value, done
            continue
        new_lines.append(line)

    while new_lines and new_lines[-1].strip() == '':
        new_lines.pop()
    if new_lines:
        new_lines[-1] = new_lines[-1].rstrip('\n') + '\n'

    with open(f, 'w') as fh:
        fh.writelines(new_lines)
" || true

# cleanup special
sed -i 's/G-Core/YATT/g' README.md
git add .

# new branch
git commit -m 'drop priv code' --no-verify

git remote add txpub git@git.woa.com:wepsdl/gcore.git
git remote add github git@github.com:Tencent/G-Core-GH.git

if [[ "$MODE" == "append" ]]; then
    # fetch txpub/public as base, apply diff as a single new commit
    git fetch txpub public
    git branch -D public 2>/dev/null || true
    git checkout -b public txpub/public
    git diff txpub/public prepare_public | git apply --index --allow-empty
    git add -A
    git commit -m 'ci: update public' --author 'mmbaseplt2<mmbaseplt2@tencent.com>' --no-verify

    # push
    git push txpub public
    git push -f github public
else
    # overwrite: discard public history, single fresh commit
    git branch -D public 2>/dev/null || true
    git checkout --orphan public
    git add -A
    git commit -m 'ci: update public' --author 'mmbaseplt2<mmbaseplt2@tencent.com>' --no-verify

    # force-push to wipe remote history
    git push -f txpub public
    git push -f github public
fi

git remote remove txpub
git remote remove github
git checkout master
git branch -D prepare_public
git branch -D public
