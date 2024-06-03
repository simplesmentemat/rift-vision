import cv2


class TemplateLoader:
    def __init__(self, minimap_template_path, gold_template_path):
        self.minimap_template_path = minimap_template_path
        self.gold_template_path = gold_template_path

    def load_template(self, template_path):
        template = cv2.imread(template_path)
        if template is None:
            raise IOError(f"Template image not found at {template_path}.")
        return template

    def load_templates(self):
        minimap_template = self.load_template(self.minimap_template_path)
        gold_template = self.load_template(self.gold_template_path)
        return minimap_template, gold_template
