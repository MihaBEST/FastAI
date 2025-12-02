#!/usr/bin/env python
"""
Запуск всех тестов библиотеки FastAI
"""

import sys
import os

# Добавляем src в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_all_tests():
    """Запускает все тесты"""
    print("=" * 60)
    print("Запуск всех тестов FastAI")
    print("=" * 60)

    # Импортируем и запускаем все тестовые модули
    test_modules = [
        'test_kneighbors',
        'test_decision_tree',
        'test_text_classifier',
        'test_basic_network',
        'test_cnn_network',
        'test_clusters',
        'test_image_classifier',
        'test_sound_analiser',
        'test_eazy_models'
        #'test_prepare_data',
        #'test_utils'
    ]

    total_passed = 0
    total_failed = 0

    for module_name in test_modules:
        try:
            module = __import__(module_name)
            print(f"\nТестирование {module_name}...")

            if hasattr(module, 'run_tests'):
                passed, failed = module.run_tests()
                total_passed += passed
                total_failed += failed
            else:
                print(f"  ⚠  Модуль {module_name} не имеет функции run_tests")

        except ImportError as e:
            print(f"  ❌ Ошибка импорта {module_name}: {e}")
        except Exception as e:
            print(f"  ❌ Ошибка в {module_name}: {e}")

    print("\n" + "=" * 60)
    print("ИТОГ:")
    print(f"  Успешно: {total_passed}")
    print(f"  Провалено: {total_failed}")
    print(f"  Всего тестов: {total_passed + total_failed}")

    if total_failed == 0:
        print("  ✅ Все тесты пройдены успешно!")
        return True
    else:
        print(f"  ❌ Найдено {total_failed} ошибок")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)