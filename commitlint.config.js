module.exports = {
  extends: ['@commitlint/config-conventional'],
  rules: {
    'type-enum': [
      2,
      'always',
      [
        'feat', // New feature
        'fix', // Bug fix
        'docs', // Documentation
        'style', // Formatting, missing semi colons, etc.
        'refactor', // Code change that neither fixes a bug nor adds a feature
        'perf', // Performance improvement
        'test', // Adding missing tests
        'chore', // Changes to the build process or auxiliary tools
        'ci', // CI/CD changes
        'build', // Build system or external dependencies
        'revert', // Reverting changes
      ],
    ],
    'subject-max-length': [2, 'always', 100],
  },
};
