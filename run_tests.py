#!/usr/bin/env python
"""
Скрипт для запуска всех тестов библиотеки FastAI
"""

import os
import sys


def main():
    # Добавляем текущую директорию в путь Python
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Запускаем все тесты через test_all.py
    from tests.test_all import run_all_tests

    print("FastAI Library Tests")
    print("=" * 60)

    success = run_all_tests()

    if success:
        print("\n✅ Все тесты пройдены успешно!")
        return 0
    else:
        print("\n❌ Найдены ошибки в тестах!")
        return 1


if __name__ == "__main__":
    sys.exit(main())