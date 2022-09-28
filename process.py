from argparse import ArgumentParser, ArgumentTypeError
import creatools


def _run_query(args, oracle):
    print()
    oracle.run(' '.join(args.terms), args.threshold, args.num_best)


def _run_multi(args, oracle):
    print()
    while query := input('Digite os termos ([Enter] para terminar): '):
        oracle.run(query, args.threshold, args.num_best)


def _parse_args():
    def check_positive(n):
        n = int(n)
        if n <= 0:
            raise ArgumentTypeError(f'n={n} <= 0')
        return n

    def check_threshold(t):
        t = float(t)
        if not (-1.0 <= t <= 1.0):
            raise ArgumentTypeError(f't={t} ∉ [-1.0, 1.0]')
        return t

    parser = ArgumentParser(description='Busca de termos em ementas',
                            add_help=False)

    subparsers = parser.add_subparsers(help='sub-comandos')

    query = subparsers.add_parser('query', help='comando para query única')
    query.add_argument('degree', help='nome do curso')
    query.add_argument('terms', nargs='+', help='termos')
    query.set_defaults(func=_run_query)

    multi = subparsers.add_parser('multi', help='comando para múltiplas queries')
    multi.add_argument('degree', help='nome do curso')
    multi.set_defaults(func=_run_multi)

    parser.add_argument('-h', '--help', action='help',
                        help='mostra esta mensagem de ajuda e termina o programa.')
    parser.add_argument('-n', '--num_best', type=check_positive, default=5,
                        help='número máximo de tópicos a apresentar')
    parser.add_argument('-t', '--threshold', type=check_threshold, default=0.0,
                        help='valor mínimo de similaridade aceito [-1, 1]')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='modo verboso')

    return parser.parse_args()


def main():
    args = _parse_args()
    oracle = creatools.Oracle(args.degree, creatools.NLPPreprocessor.pt(),
                              creatools.Models.LsiModel, args.verbose)
    args.func(args, oracle)


if __name__ == '__main__':
    main()
