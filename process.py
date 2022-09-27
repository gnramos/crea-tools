from argparse import ArgumentParser, ArgumentTypeError
import creatools


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
    parser.add_argument('-h', '--help', action='help',
                        help='mostra esta mensagem de ajuda e termina o programa.')
    parser.add_argument('degree', help='nome do curso')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q', '--query', nargs='+',
                       help='query única (fornecida como argumento)')
    group.add_argument('-m', '--multi_query', action='store_true',
                       help='múltiplas queries (recebidas iterativamente)')

    parser.add_argument('-n', '--num_best', type=check_positive, default=5,
                        help='número máximo de tópicos a apresentar')
    parser.add_argument('-t', '--threshold', type=check_threshold, default=0.0,
                        help='valor mínimo de similaridade aceito [-1, 1]')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='modo verboso')

    return parser.parse_args()


def main():
    args = _parse_args()
    oracle = creatools.Oracle(args.degree, creatools.Preprocessor.pt(),
                              creatools.Models.LsiModel, args.verbose)

    if args.multi_query:
        while query := input('Digite os termos ([Enter] para terminar): '):
            oracle.run(query)
    else:
        oracle.run(' '.join(args.query))


if __name__ == '__main__':
    main()
