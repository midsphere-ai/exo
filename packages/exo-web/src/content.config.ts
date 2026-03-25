import { defineCollection, z } from "astro:content";

const docs = defineCollection({
  type: "content",
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    section: z.enum([
      "getting-started",
      "guides",
      "api-reference",
      "concepts",
    ]),
    order: z.number().default(0),
  }),
});

export const collections = { docs };
