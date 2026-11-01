"""Full DPO from matching SFT checkpoints using the standard torch loss.

TRL is absent in the fixed environment. lab.training implements the usual
-beta-scaled log-sigmoid of policy minus frozen-reference log-ratios directly.
"""
from lab.scripts.sft import _main


def main(argv=None):
    return _main(argv, 'dpo')


if __name__ == '__main__':
    raise SystemExit(main())
