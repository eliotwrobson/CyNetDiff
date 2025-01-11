envdata = environment()
python_path = [meson.project_build_root(), meson.project_source_root()]
envdata.append('PYTHONPATH', python_path)

test(
  'test-suite',
  python,
  args: ['-m', 'pytest'],
  workdir: meson.project_source_root(),
  env: envdata,
  timeout: 90
)
