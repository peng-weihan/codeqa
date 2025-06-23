#!/usr/bin/env python
import os
import sys
import json
import logging
import asyncio
from pathlib import Path


# 设置项目路径以便能够导入本地模块
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('code_aot_run.log')
    ]
)
logger = logging.getLogger("code_aot")

# pylint: disable=wrong-import-position
from code_aot.atom import atom_with_context
# from utils.code_formatting import format_code

async  def main():
    question_example = {
    "question": "What is the purpose of the `get_accessed_time` in `FileSystemStorage`?",
    "answer": "",
    "relative_code_list": [
      {
        "start_line": 19,
        "end_line": 228,
        "belongs_to": {
          "file_name": "filesystem.py",
          "upper_path": "/Users/xinyun/Programs/django/django/core/files/storage",
          "module": "storage",
          "define_class": [
            "FileSystemStorage"
          ],
          "imports": [
            "os",
            "datetime.UTC",
            "datetime.datetime",
            "urllib.parse.urljoin",
            "django.conf.settings",
            "django.core.files.File",
            "django.core.files.locks",
            "django.core.files.move.file_move_safe",
            "django.core.signals.setting_changed",
            "django.utils._os.safe_join",
            "django.utils.deconstruct.deconstructible",
            "django.utils.encoding.filepath_to_uri",
            "django.utils.functional.cached_property",
            "base.Storage",
            "mixins.StorageSettingsMixin"
          ]
        },
        "relative_function": [],
        "code": "class FileSystemStorage(Storage, StorageSettingsMixin):\n    \"\"\"\n    Standard filesystem storage\n    \"\"\"\n\n    def __init__(\n        self,\n        location=None,\n        base_url=None,\n        file_permissions_mode=None,\n        directory_permissions_mode=None,\n        allow_overwrite=False,\n    ):\n        self._location = location\n        self._base_url = base_url\n        self._file_permissions_mode = file_permissions_mode\n        self._directory_permissions_mode = directory_permissions_mode\n        self._allow_overwrite = allow_overwrite\n        setting_changed.connect(self._clear_cached_properties)\n\n    @cached_property\n    def base_location(self):\n        return self._value_or_setting(self._location, settings.MEDIA_ROOT)\n\n    @cached_property\n    def location(self):\n        return os.path.abspath(self.base_location)\n\n    @cached_property\n    def base_url(self):\n        if self._base_url is not None and not self._base_url.endswith(\"/\"):\n            self._base_url += \"/\"\n        return self._value_or_setting(self._base_url, settings.MEDIA_URL)\n\n    @cached_property\n    def file_permissions_mode(self):\n        return self._value_or_setting(\n            self._file_permissions_mode, settings.FILE_UPLOAD_PERMISSIONS\n        )\n\n    @cached_property\n    def directory_permissions_mode(self):\n        return self._value_or_setting(\n            self._directory_permissions_mode, settings.FILE_UPLOAD_DIRECTORY_PERMISSIONS\n        )\n\n    def _open(self, name, mode=\"rb\"):\n        return File(open(self.path(name), mode))\n\n    def _save(self, name, content):\n        full_path = self.path(name)\n\n        # Create any intermediate directories that do not exist.\n        directory = os.path.dirname(full_path)\n        try:\n            if self.directory_permissions_mode is not None:\n                # Set the umask because os.makedirs() doesn't apply the \"mode\"\n                # argument to intermediate-level directories.\n                old_umask = os.umask(0o777 & ~self.directory_permissions_mode)\n                try:\n                    os.makedirs(\n                        directory, self.directory_permissions_mode, exist_ok=True\n                    )\n                finally:\n                    os.umask(old_umask)\n            else:\n                os.makedirs(directory, exist_ok=True)\n        except FileExistsError:\n            raise FileExistsError(\"%s exists and is not a directory.\" % directory)\n\n        # There's a potential race condition between get_available_name and\n        # saving the file; it's possible that two threads might return the\n        # same name, at which point all sorts of fun happens. So we need to\n        # try to create the file, but if it already exists we have to go back\n        # to get_available_name() and try again.\n\n        while True:\n            try:\n                # This file has a file path that we can move.\n                if hasattr(content, \"temporary_file_path\"):\n                    file_move_safe(\n                        content.temporary_file_path(),\n                        full_path,\n                        allow_overwrite=self._allow_overwrite,\n                    )\n\n                # This is a normal uploadedfile that we can stream.\n                else:\n                    # The combination of O_CREAT and O_EXCL makes os.open() raises an\n                    # OSError if the file already exists before it's opened.\n                    open_flags = (\n                        os.O_WRONLY\n                        | os.O_CREAT\n                        | os.O_EXCL\n                        | getattr(os, \"O_BINARY\", 0)\n                    )\n                    if self._allow_overwrite:\n                        open_flags = open_flags & ~os.O_EXCL | os.O_TRUNC\n                    fd = os.open(full_path, open_flags, 0o666)\n                    _file = None\n                    try:\n                        locks.lock(fd, locks.LOCK_EX)\n                        for chunk in content.chunks():\n                            if _file is None:\n                                mode = \"wb\" if isinstance(chunk, bytes) else \"wt\"\n                                _file = os.fdopen(fd, mode)\n                            _file.write(chunk)\n                    finally:\n                        locks.unlock(fd)\n                        if _file is not None:\n                            _file.close()\n                        else:\n                            os.close(fd)\n            except FileExistsError:\n                # A new name is needed if the file exists.\n                name = self.get_available_name(name)\n                full_path = self.path(name)\n            else:\n                # OK, the file save worked. Break out of the loop.\n                break\n\n        if self.file_permissions_mode is not None:\n            os.chmod(full_path, self.file_permissions_mode)\n\n        # Ensure the saved path is always relative to the storage root.\n        name = os.path.relpath(full_path, self.location)\n        # Ensure the moved file has the same gid as the storage root.\n        self._ensure_location_group_id(full_path)\n        # Store filenames with forward slashes, even on Windows.\n        return str(name).replace(\"\\\\\", \"/\")\n\n    def _ensure_location_group_id(self, full_path):\n        if os.name == \"posix\":\n            file_gid = os.stat(full_path).st_gid\n            location_gid = os.stat(self.location).st_gid\n            if file_gid != location_gid:\n                try:\n                    os.chown(full_path, uid=-1, gid=location_gid)\n                except PermissionError:\n                    pass\n\n    def delete(self, name):\n        if not name:\n            raise ValueError(\"The name must be given to delete().\")\n        name = self.path(name)\n        # If the file or directory exists, delete it from the filesystem.\n        try:\n            if os.path.isdir(name):\n                os.rmdir(name)\n            else:\n                os.remove(name)\n        except FileNotFoundError:\n            # FileNotFoundError is raised if the file or directory was removed\n            # concurrently.\n            pass\n\n    def is_name_available(self, name, max_length=None):\n        if self._allow_overwrite:\n            return not (max_length and len(name) > max_length)\n        return super().is_name_available(name, max_length=max_length)\n\n    def get_alternative_name(self, file_root, file_ext):\n        if self._allow_overwrite:\n            return f\"{file_root}{file_ext}\"\n        return super().get_alternative_name(file_root, file_ext)\n\n    def exists(self, name):\n        return os.path.lexists(self.path(name))\n\n    def listdir(self, path):\n        path = self.path(path)\n        directories, files = [], []\n        with os.scandir(path) as entries:\n            for entry in entries:\n                if entry.is_dir():\n                    directories.append(entry.name)\n                else:\n                    files.append(entry.name)\n        return directories, files\n\n    def path(self, name):\n        return safe_join(self.location, name)\n\n    def size(self, name):\n        return os.path.getsize(self.path(name))\n\n    def url(self, name):\n        if self.base_url is None:\n            raise ValueError(\"This file is not accessible via a URL.\")\n        url = filepath_to_uri(name)\n        if url is not None:\n            url = url.lstrip(\"/\")\n        return urljoin(self.base_url, url)\n\n    def _datetime_from_timestamp(self, ts):\n        \"\"\"\n        If timezone support is enabled, make an aware datetime object in UTC;\n        otherwise make a naive one in the local timezone.\n        \"\"\"\n        tz = UTC if settings.USE_TZ else None\n        return datetime.fromtimestamp(ts, tz=tz)\n\n    def get_accessed_time(self, name):\n        return self._datetime_from_timestamp(os.path.getatime(self.path(name)))\n\n    def get_created_time(self, name):\n        return self._datetime_from_timestamp(os.path.getctime(self.path(name)))\n\n    def get_modified_time(self, name):\n        return self._datetime_from_timestamp(os.path.getmtime(self.path(name)))"
      },
      {
        "start_line": 221,
        "end_line": 222,
        "belongs_to": {
          "file_name": "filesystem.py",
          "upper_path": "/Users/xinyun/Programs/django/django/core/files/storage",
          "module": "storage",
          "define_class": [
            "FileSystemStorage"
          ],
          "imports": [
            "os",
            "datetime.UTC",
            "datetime.datetime",
            "urllib.parse.urljoin",
            "django.conf.settings",
            "django.core.files.File",
            "django.core.files.locks",
            "django.core.files.move.file_move_safe",
            "django.core.signals.setting_changed",
            "django.utils._os.safe_join",
            "django.utils.deconstruct.deconstructible",
            "django.utils.encoding.filepath_to_uri",
            "django.utils.functional.cached_property",
            "base.Storage",
            "mixins.StorageSettingsMixin"
          ]
        },
        "relative_function": [],
        "code": "def get_accessed_time(self, name):\n        return self._datetime_from_timestamp(os.path.getatime(self.path(name)))"
      }
    ],
    "score": ""
    }
    # code_prompt = format_code(question_example["relative_code_list"])
    return await atom_with_context(question = question_example["question"], contexts=question_example["relative_code_list"])


if __name__ == "__main__":
    res = asyncio.run(main())
    print(res)
